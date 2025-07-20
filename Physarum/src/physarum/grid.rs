use std::fmt::{Display, Formatter};

use super::blur::Blur;
use super::population_config::PopulationConfig;
use rand::Rng;

use rand::distributions::Uniform;
use rand::rngs::SmallRng;

pub struct Grid {
    pub config: PopulationConfig,
    pub width: usize,
    pub height: usize,
    data: Vec<f32>,
    buf: Vec<f32>,
    blur: Blur,
}

impl Grid {
    pub fn new(width: usize, height: usize, config: PopulationConfig, rng: &mut SmallRng) -> Self {
        if !width.is_power_of_two() || !height.is_power_of_two() {
            panic!("Grid dims must be 2^n");
        }
        let range = Uniform::from(0.0..1.0);
        let data = rng.sample_iter(range).take(width * height).collect();

        Grid {
            width,
            height,
            data,
            config,
            buf: vec![0.0; width * height],
            blur: Blur::new(width),
        }
    }

    #[inline(always)]
    fn index(&self, x: f32, y: f32) -> usize {
        let i = (x + self.width as f32) as usize & (self.width - 1);
        let j = (y + self.height as f32) as usize & (self.height - 1);
        j * self.width + i
    }

    #[inline(always)]
    pub fn get_buf(&self, x: f32, y: f32) -> f32 {
        self.buf[self.index(x, y)]
    }

    #[inline(always)]
    pub fn deposit(&mut self, x: f32, y: f32) {
        let idx = self.index(x, y);
        self.data[idx] += self.config.deposition_amount;
    }

    pub fn diffuse(&mut self, radius: f32) {
        self.blur.run(
            &mut self.data,
            &mut self.buf,
            self.width,
            self.height,
            radius,
            self.config.decay_factor,
        );
    }

    pub fn quantile(&self, fraction: f32) -> f32 {
        let index = if (fraction - 1.0_f32).abs() < f32::EPSILON {
            self.data.len() - 1
        } else {
            (self.data.len() as f32 * fraction) as usize
        };
        let mut sorted = self.data.clone();
        sorted.as_mut_slice().select_nth_unstable_by(index, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted[index]
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

impl Display for PopulationConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{\n  Sensor Distance: {},\n  Step Distance: {},\n  Sensor Angle: {},\n  Rotation Angle: {},\n  Decay Factor: {},\n  Deposition Amount: {},\n}}",
            self.sensor_distance,
            self.step_distance,
            self.sensor_angle,
            self.rotation_angle,
            self.decay_factor,
            self.deposition_amount
        )
    }
}

pub fn combine<T>(grids: &mut [Grid], attraction_table: &[T])
where
    T: AsRef<[f32]> + Sync,
{
    let datas: Vec<_> = grids.iter().map(|grid| &grid.data).collect();
    let bufs: Vec<_> = grids.iter().map(|grid| &grid.buf).collect();
    bufs.iter().enumerate().for_each(|(i, buf)| unsafe {
        let buf_ptr = *buf as *const Vec<f32> as *mut Vec<f32>;
        buf_ptr.as_mut().unwrap().fill(0.0);
        datas.iter().enumerate().for_each(|(j, other)| {
            let multiplier = attraction_table[i].as_ref()[j];
            buf_ptr
                .as_mut()
                .unwrap()
                .iter_mut()
                .zip(*other)
                .for_each(|(to, from)| *to += from * multiplier)
        })
    });
}

use super::grid;
use super::grid::Grid;
use super::palette;
use super::palette::Palette;
use super::particle::Particle;
use super::population_config::PopulationConfig;
use nannou::image::{DynamicImage, Rgba};
use once_cell::sync::Lazy;
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::ParallelSliceMut;
use rayon::{
    iter::ParallelBridge,
    prelude::{IntoParallelRefMutIterator, ParallelIterator},
};
pub struct PhysarumModel {
    pub grids: Vec<Grid>,
    agents: Vec<Particle>,
    attraction_table: Vec<Vec<f32>>,
    diffusity: f32,
    iteration: i32,
    palette: Palette,
}

impl PhysarumModel {
    const ATTRACTION_FACTOR_MEAN: f32 = 1.0;
    const ATTRACTION_FACTOR_STD: f32 = 0.1;
    const REPULSION_FACTOR_MEAN: f32 = -1.0;
    const REPULSION_FACTOR_STD: f32 = 0.1;

    pub fn new(
        width: usize,
        height: usize,
        n_particles: usize,
        n_populations: usize,
        diffusity: f32,
        palette_index: usize,
        rng: &mut SmallRng,
    ) -> Self {
        let particles_per_grid = (n_particles as f64 / n_populations as f64).ceil() as usize;
        let n_particles = particles_per_grid * n_populations;

        let attraction_distr =
            Normal::new(Self::ATTRACTION_FACTOR_MEAN, Self::ATTRACTION_FACTOR_STD).unwrap();
        let repulstion_distr =
            Normal::new(Self::REPULSION_FACTOR_MEAN, Self::REPULSION_FACTOR_STD).unwrap();

        let mut attraction_table = Vec::with_capacity(n_populations);
        for i in 0..n_populations {
            attraction_table.push(Vec::with_capacity(n_populations));
            for j in 0..n_populations {
                attraction_table[i].push(if i == j {
                    attraction_distr.sample(rng)
                } else {
                    repulstion_distr.sample(rng)
                });
            }
        }

        PhysarumModel {
            agents: (0..n_particles)
                .map(|i| Particle::new(width, height, i / particles_per_grid, rng))
                .collect(),
            grids: (0..n_populations)
                .map(|_| Grid::new(width, height, PopulationConfig::new(rng), rng))
                .collect(),
            attraction_table,
            diffusity,
            iteration: 0,
            palette: palette::PALETTE_ARRAY[palette_index],
        }
    }

    pub fn set_population_configs(&mut self, configs: Vec<PopulationConfig>) {
        if configs.len() < self.grids.len() {
            panic!("Expected same lenght vecs for grid and config")
        }

        self.grids.iter_mut().enumerate().for_each(|(i, grid)| {
            grid.config = configs[i];
        })
    }

    fn pick_direction(center: f32, left: f32, right: f32, rng: &mut SmallRng) -> f32 {
        if (center > left) && (center > right) {
            0.0
        } else if (center < left) && (center < right) {
            *[-1.0, 1.0].choose(rng).unwrap()
        } else if left < right {
            1.0
        } else if right < left {
            -1.0
        } else {
            0.0
        }
    }

    pub fn step_simulation_agents(&mut self) {
        let grids = &mut self.grids;
        grid::combine(grids, &self.attraction_table);

        self.agents.par_iter_mut().for_each(|agent| {
            let grid = &grids[agent.id];
            let PopulationConfig {
                sensor_distance,
                sensor_angle,
                rotation_angle,
                step_distance,
                ..
            } = grid.config;
            let (width, height) = (grid.width, grid.height);

            let xc = agent.x + agent.angle.cos() * sensor_distance;
            let yc = agent.y + agent.angle.sin() * sensor_distance;
            let xl = agent.x + (agent.angle - sensor_angle).cos() * sensor_distance;
            let yl = agent.y + (agent.angle - sensor_angle).sin() * sensor_distance;
            let xr = agent.x + (agent.angle + sensor_angle).cos() * sensor_distance;
            let yr = agent.y + (agent.angle + sensor_angle).sin() * sensor_distance;

            let trail_c = grid.get_buf(xc, yc);
            let trail_l = grid.get_buf(xl, yl);
            let trail_r = grid.get_buf(xr, yr);

            let mut rng = SmallRng::seed_from_u64(agent.id as u64);
            let direction = PhysarumModel::pick_direction(trail_c, trail_l, trail_r, &mut rng);
            agent.rotate_and_move(direction, rotation_angle, step_distance, width, height);
        });

        for agent in self.agents.iter() {
            self.grids[agent.id].deposit(agent.x, agent.y);
        }

        let diffusivity = self.diffusity;
        self.grids.iter_mut().for_each(|grid| {
            grid.diffuse(diffusivity);
        });
        self.iteration += 1;
    }

    pub fn print_configurations(&self) {
        for (i, grid) in self.grids.iter().enumerate() {
            println!("Grid {}: {}", i, grid.config);
        }
        println!("Attraction table: {:#?}", self.attraction_table);
    }

    pub fn save_to_image(&self, image: &mut DynamicImage) {
        let w = self.grids[0].width as usize;
        let h = self.grids[0].height as usize;

        let palette: Vec<[f32; 3]> = self
            .palette
            .colors
            .iter()
            .map(|c| [c.0[0] as f32, c.0[1] as f32, c.0[2] as f32])
            .collect();

        let acc = self.accumulate_color_values(w, h, palette);

        write_dynamic_image(image, w, acc);
    }

    fn accumulate_color_values(
        &self,
        width: usize,
        height: usize,
        palette: Vec<[f32; 3]>,
    ) -> Vec<[f32; 3]> {
        let pix_count = width * height;

        let grid_params: Vec<(&[f32], [f32; 3], f32)> = self
            .grids
            .iter()
            .zip(palette.into_iter())
            .map(|(grid, col)| {
                let slice = grid.data();
                let inv_max = 1.0 / (grid.quantile(0.999) * 1.5);
                (slice, col, inv_max)
            })
            .collect();

        let mut acc = vec![[0.0_f32; 3]; pix_count];

        acc.par_chunks_mut(BLOCK_SIZE)
            .enumerate()
            .for_each(|(block_idx, block)| {
                let start = block_idx * BLOCK_SIZE;
                for (j, pixel_acc) in block.iter_mut().enumerate() {
                    let idx = start + j;
                    let mut r = 0.0_f32;
                    let mut g = 0.0_f32;
                    let mut b = 0.0_f32;

                    for &(slice, [cr, cg, cb], inv_max) in &grid_params {
                        let v = slice[idx];
                        let normalized = (v * inv_max).clamp(0.0, 1.0);
                        let lut_idx = (normalized * (LUT_SIZE as f32 - 1.0)).round() as usize;
                        let t = GAMMA_LUT[lut_idx];
                        r += cr * t;
                        g += cg * t;
                        b += cb * t;
                    }

                    pixel_acc[0] = r;
                    pixel_acc[1] = g;
                    pixel_acc[2] = b;
                }
            });

        acc
    }
}

const LUT_SIZE: usize = 256;
const BLOCK_SIZE: usize = 16_384;
const INV_GAMMA: f32 = 1.0 / 2.2;

static GAMMA_LUT: Lazy<[f32; LUT_SIZE]> = Lazy::new(|| {
    let mut lut = [0.0_f32; LUT_SIZE];
    let max_idx = (LUT_SIZE - 1) as f32;
    for i in 0..LUT_SIZE {
        let x = i as f32 / max_idx;
        lut[i] = x.powf(INV_GAMMA);
    }
    lut
});

fn write_dynamic_image(image: &mut DynamicImage, w: usize, acc: Vec<[f32; 3]>) {
    let buf = image.as_mut_rgba8().unwrap();
    buf.enumerate_rows_mut().par_bridge().for_each(|(_, row)| {
        for (x, y, pixel) in row {
            let idx = y as usize * w + x as usize;
            let r = to_u8(acc[idx][0]);
            let g = to_u8(acc[idx][1]);
            let b = to_u8(acc[idx][2]);
            *pixel = Rgba([r, g, b, 255]);
        }
    });
}

const fn to_u8(v: f32) -> u8 {
    if v < 0.0 {
        0u8
    } else if v > 255.0 {
        255u8
    } else {
        v as u8
    }
}

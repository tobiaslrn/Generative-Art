use itertools::multizip;

pub struct Blur {
    row_buffer: Vec<f32>,
}

impl Blur {
    pub fn new(width: usize) -> Self {
        Blur {
            row_buffer: vec![0.0; width],
        }
    }

    pub fn run(
        &mut self,
        src: &mut [f32],
        buf: &mut [f32],
        width: usize,
        height: usize,
        sigma: f32,
        decay: f32,
    ) {
        let boxes = Blur::boxes_for_gaussian::<2>(sigma);
        self.box_blur(src, buf, width, height, boxes[0], 1.0);
        self.box_blur(src, buf, width, height, boxes[1], decay);
    }

    fn boxes_for_gaussian<const N: usize>(sigma: f32) -> [usize; N] {
        let w_ideal = (12.0 * sigma * sigma / N as f32 + 1.0).sqrt();
        let mut w = w_ideal as usize;
        w -= 1 - (w & 1);
        let mut m = 0.25 * (N * (w + 3)) as f32;
        m -= 3.0 * sigma * sigma / (w + 1) as f32;
        let m = m.round() as usize;

        let mut result = [0; N];
        for (i, value) in result.iter_mut().enumerate() {
            *value = (if i < m { w - 1 } else { w + 1 }) / 2;
        }
        result
    }

    fn box_blur(
        &mut self,
        src: &mut [f32],
        buf: &mut [f32],
        width: usize,
        height: usize,
        radius: usize,
        decay: f32,
    ) {
        self.box_blur_h(src, buf, width, radius);
        self.box_blur_v(buf, src, width, height, radius, decay);
    }

    fn box_blur_h(&mut self, src: &[f32], dst: &mut [f32], width: usize, radius: usize) {
        let weight = 1.0 / (2 * radius + 1) as f32;

        src.chunks_exact(width)
            .zip(dst.chunks_exact_mut(width))
            .for_each(|(src_row, dst_row)| {
                let mut value = src_row[width - radius - 1];
                for j in 0..radius {
                    value += src_row[width - radius + j] + src_row[j];
                }

                for (i, dst_elem) in dst_row.iter_mut().enumerate() {
                    let left = (i + width - radius - 1) & (width - 1);
                    let right = (i + radius) & (width - 1);
                    value += src_row[right] - src_row[left];
                    *dst_elem = value * weight;
                }
            })
    }

    fn box_blur_v(
        &mut self,
        src: &[f32],
        dst: &mut [f32],
        width: usize,
        height: usize,
        radius: usize,
        decay: f32,
    ) {
        let weight = decay / (2 * radius + 1) as f32;
        let offset = (height - radius - 1) * width;
        self.row_buffer
            .copy_from_slice(&src[offset..offset + width]);

        for j in 0..radius {
            let bottom_off = (height - radius + j) * width;
            let bottom_row = &src[bottom_off..bottom_off + width];
            let top_off = j * width;
            let top_row = &src[top_off..top_off + width];

            for (buf, bottom, top) in multizip((&mut self.row_buffer, bottom_row, top_row)) {
                *buf += bottom + top;
            }
        }

        for (i, dst_row) in dst.chunks_exact_mut(width).enumerate() {
            let bottom_off = ((i + height - radius - 1) & (height - 1)) * width;
            let bottom_row = &src[bottom_off..bottom_off + width];
            let top_off = ((i + radius) & (height - 1)) * width;
            let top_row = &src[top_off..top_off + width];

            for (dst, buf, bottom, top) in
                multizip((dst_row, &mut self.row_buffer, bottom_row, top_row))
            {
                *buf += top - bottom;
                *dst = *buf * weight;
            }
        }
    }
}

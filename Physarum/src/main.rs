#![allow(dead_code)]
pub mod physarum;
pub mod ui;

use chrono::{Datelike, Timelike};
use fps_ticker::Fps;
use nannou::{image::DynamicImage, prelude::*, wgpu::Texture};
use nannou_egui::{self, Egui};
use physarum::physarum_model::PhysarumModel;
use physarum::population_config::PopulationConfig;
use rand::prelude::*;

fn main() {
    nannou::app(model).update(update).run();
}

struct PhysarumSettings {
    model: PhysarumModel,
    configs: Vec<PopulationConfig>,
    config_changed: bool,
    width: usize,
    height: usize,
    n_particles: usize,
    n_populations: usize,
    diffusivity: f32,
    palette_idx: usize,
}

pub struct Model {
    physarum_settings: PhysarumSettings,
    fps_counter: Fps,
    seed: u64,
    egui_visible: bool,
    steps_per_frame: usize,
    egui: Egui,
    main_window_id: WindowId,
    rng: SmallRng,
    changed: bool,
    render: bool,
    image: DynamicImage,
}

impl Model {
    fn new(
        physarum_settings: PhysarumSettings,
        seed: u64,
        rng: SmallRng,
        egui: Egui,
        main_window_id: WindowId,
        image: DynamicImage,
    ) -> Self {
        Model {
            physarum_settings,
            fps_counter: Fps::with_window_len(60),
            steps_per_frame: 1,
            egui_visible: true,
            changed: true,
            render: false,
            main_window_id,
            seed,
            egui,
            rng,
            image,
        }
    }

    fn reset_rng(&mut self) {
        self.rng = SmallRng::seed_from_u64(self.seed);
    }

    fn reset(&mut self) {
        self.reset_rng();
        let sc = &self.physarum_settings;
        let physarum_model = PhysarumModel::new(
            sc.width,
            sc.height,
            sc.n_particles,
            sc.n_populations,
            sc.diffusivity,
            sc.palette_idx,
            &mut self.rng,
        );
        self.physarum_settings.model = physarum_model;
        self.physarum_settings
            .model
            .set_population_configs(self.physarum_settings.configs.clone());
    }
}

pub fn get_random_configs(rng: &mut SmallRng) -> Vec<PopulationConfig> {
    let mut configs = vec![];
    (0..5).for_each(|_| {
        configs.push(PopulationConfig::new(rng));
    });
    configs
}

fn model(app: &App) -> Model {
    let seed_set: Option<u64> = Some(0);
    let config_set: Option<String> = Some(
        r#"[{"sensor_distance":49.44998,"step_distance":1.4334296,"sensor_angle":1.4103054,"rotation_angle":1.3992796,"decay_factor":0.1,"deposition_amount":5.0},{"sensor_distance":16.194483,"step_distance":1.0892088,"sensor_angle":0.6044981,"rotation_angle":1.3636005,"decay_factor":0.1,"deposition_amount":5.0},{"sensor_distance":49.455574,"step_distance":0.92719734,"sensor_angle":2.0906055,"rotation_angle":0.57098943,"decay_factor":0.1,"deposition_amount":5.0},{"sensor_distance":25.101528,"step_distance":1.8175551,"sensor_angle":1.9417597,"rotation_angle":0.30816564,"decay_factor":0.1,"deposition_amount":5.0},{"sensor_distance":51.51055,"step_distance":1.4250815,"sensor_angle":0.14544931,"rotation_angle":1.485748,"decay_factor":0.1,"deposition_amount":5.0}]"#.to_string(),
    );

    let image_window = app
        .new_window()
        .size(1400, 1024)
        .view(view)
        .key_pressed(key_pressed)
        .mouse_pressed(mouse_pressed)
        .raw_event(raw_window_event)
        .title("Physarum")
        .build()
        .unwrap();

    let window = app.window(image_window).unwrap();
    let seed = seed_set.unwrap_or(thread_rng().next_u64());
    let mut rng = SmallRng::seed_from_u64(seed);
    println!("{}", seed);

    // Physarum
    let (width, height) = (1024, 1024);
    let n_particles = 200000;
    let n_populations = 2;
    let diffusivity = 1.0;
    let palette_idx = 0;
    let configs: Vec<PopulationConfig> = {
        match config_set {
            Some(val) => serde_json::from_str(&val).unwrap_or(get_random_configs(&mut rng)),
            None => get_random_configs(&mut rng),
        }
    };
    let physarum_model = PhysarumModel::new(
        width,
        height,
        n_particles,
        n_populations,
        diffusivity,
        palette_idx,
        &mut rng,
    );

    let physarum = PhysarumSettings {
        width,
        height,
        n_particles,
        n_populations,
        diffusivity,
        model: physarum_model,
        palette_idx,
        configs,
        config_changed: true,
    };

    Model::new(
        physarum,
        seed,
        rng,
        Egui::from_window(&window),
        image_window,
        DynamicImage::new_rgba8(width as u32, height as u32),
    )
}

fn update(_app: &App, model: &mut Model, update: Update) {
    model.fps_counter.tick();
    model.changed = false;
    if model.egui_visible {
        ui::update_gui(model, update);
    }
    for _ in 0..model.steps_per_frame {
        model.physarum_settings.model.step_simulation_agents();
    }
    model
        .physarum_settings
        .model
        .save_to_image(&mut model.image);
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);
    let texture = Texture::from_image(app, &model.image);
    let tx_w = texture.size()[1] as f32 * 0.5;
    let right = app.window_rect().right();
    let offset = right - tx_w;
    draw.texture(&texture).x(offset);
    draw.to_frame(app, &frame).unwrap();
    if model.egui_visible {
        model.egui.draw_to_frame(&frame).unwrap();
    }
    if model.render {
        let _ = &model
            .image
            .save_with_format(get_path(app), nannou::image::ImageFormat::Png);
    }
}

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    model.egui.handle_raw_event(event);
}

fn key_pressed(app: &App, model: &mut Model, key: Key) {
    if key == Key::S {
        save_frame(app);
    } else if key == Key::F {
        model.egui_visible = !model.egui_visible;
    } else if key == Key::D {
        model.physarum_settings.model.print_configurations();
    }
}

fn mouse_pressed(_app: &App, _model: &mut Model, button: MouseButton) {
    if button == MouseButton::Left {}
}

fn save_frame(app: &App) {
    let path = get_path(app);
    app.main_window().capture_frame(path);
}

fn get_path(app: &App) -> String {
    let now = chrono::offset::Local::now();
    let time = format!(
        "{}_{}_{}_{}_{}_{}",
        now.year(),
        now.month(),
        now.day(),
        now.hour(),
        now.minute(),
        now.second()
    );
    let path = format!(
        "{}/{}_{}{}",
        "renders",
        app.exe_name().unwrap(),
        time,
        ".png"
    );
    path
}

//! Resident multi-request Qwen-Image demo-loop driver (EngineTailor round 31).
//!
//! Loads the model once in this process, then serves a JSON spec of image
//! requests back to back — the resident/demo form factor, as opposed to the
//! one-shot `generate-image` CLI, which pays one process and one model load
//! per request. The engine's own `APXINF_QWEN_IMAGE_TIMING=1` lines carry the
//! per-request phase timings; this driver adds per-request `qimage-loop`
//! markers and a sequence total so an A/B runner can pair requests across
//! arms. It changes no engine semantics: it calls the same public
//! `AutoModel`/`ImageRequest` path as the CLI, with the same process-wide
//! MLX cache limit the image CLI sets.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --features coreml --example qwen_image_loop -- \
//!   --model <checkpoint dir> --spec <spec.json> --output-dir <dir> \
//!   [--model-variant auto]
//! ```
//!
//! Spec JSON:
//!
//! ```json
//! {"requests": [
//!   {"name": "r1", "prompt_file": "prompt.txt", "width": 512, "height": 512,
//!    "seed": 44, "steps": 40, "step_cache_ratio": 0.25},
//!   {"name": "r2", "prompt": "inline prompt", "width": 512, "height": 512,
//!    "seed": 45, "steps": 40, "step_cache_ratio": null}
//! ]}
//! ```
//!
//! Exactly one of `prompt` / `prompt_file` is required; file contents are
//! trimmed like the A/B runners do. `step_cache_ratio: null` keeps the family
//! default. Each request writes `<output-dir>/<name>.ppm`.

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod native {
    use apxinf_core::Device;
    use apxinf_model::{AutoModel, ImageRequest, LoadOptions};
    use serde::Deserialize;
    use std::path::PathBuf;
    use std::time::Instant;

    #[derive(Deserialize)]
    struct Spec {
        requests: Vec<Entry>,
    }

    #[derive(Deserialize)]
    struct Entry {
        name: String,
        prompt: Option<String>,
        prompt_file: Option<PathBuf>,
        width: usize,
        height: usize,
        seed: u64,
        steps: usize,
        step_cache_ratio: Option<f32>,
    }

    const USAGE: &str = "usage: qwen_image_loop --model <dir> --spec <json> \
                         --output-dir <dir> [--model-variant auto]";

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut model_dir = None;
        let mut model_variant = None;
        let mut spec_path = None;
        let mut out_dir = None;
        let mut i = 0;
        while i < args.len() {
            let value = args
                .get(i + 1)
                .ok_or_else(|| format!("missing value for {}", args[i]))?;
            match args[i].as_str() {
                "--model" => model_dir = Some(PathBuf::from(value)),
                "--model-variant" => model_variant = Some(value.clone()),
                "--spec" => spec_path = Some(PathBuf::from(value)),
                "--output-dir" => out_dir = Some(PathBuf::from(value)),
                other => return Err(format!("unknown argument {other}\n{USAGE}").into()),
            }
            i += 2;
        }
        let (model_dir, spec_path, out_dir) = match (model_dir, spec_path, out_dir) {
            (Some(m), Some(s), Some(o)) => (m, s, o),
            _ => return Err(USAGE.into()),
        };
        let spec: Spec = serde_json::from_slice(&std::fs::read(&spec_path)?)?;
        if spec.requests.is_empty() {
            return Err("spec has no requests".into());
        }

        // The image CLI's process-wide allocator policy, replicated so the
        // resident form matches the one-shot form's memory behavior.
        apxinf_mlx::set_cache_limit(1_000_000_000)?;
        apxinf_mlx::clear_cache()?;

        let options = LoadOptions {
            model_variant,
            ..LoadOptions::default()
        };
        eprintln!("qimage-loop: loading from {} ...", model_dir.display());
        let started = Instant::now();
        let mut model = AutoModel::load_model(Device::Metal(0), &model_dir, &options)?;
        eprintln!(
            "qimage-loop: loaded in {:.3}s requests={}",
            started.elapsed().as_secs_f64(),
            spec.requests.len()
        );

        std::fs::create_dir_all(&out_dir)?;
        let sequence = Instant::now();
        for entry in &spec.requests {
            if entry.name.is_empty() || entry.name.contains(['/', '\\']) {
                return Err(format!("bad request name {:?}", entry.name).into());
            }
            let file_prompt;
            let prompt = match (&entry.prompt, &entry.prompt_file) {
                (Some(prompt), None) => prompt.as_str(),
                (None, Some(path)) => {
                    file_prompt = std::fs::read_to_string(path)?;
                    file_prompt.trim()
                }
                _ => {
                    return Err(format!(
                        "request {:?} needs exactly one of prompt / prompt_file",
                        entry.name
                    )
                    .into())
                }
            };
            let request = ImageRequest {
                prompt,
                width: entry.width,
                height: entry.height,
                seed: entry.seed,
                steps: entry.steps,
                step_cache_ratio: entry.step_cache_ratio,
            };
            let started = Instant::now();
            let image = model.generate_image(&request)?;
            eprintln!(
                "qimage-loop: {} generated in {:.3}s",
                entry.name,
                started.elapsed().as_secs_f64()
            );
            let mut ppm =
                format!("P6\n{} {}\n255\n", image.width, image.height).into_bytes();
            ppm.extend_from_slice(&image.rgb);
            let path = out_dir.join(format!("{}.ppm", entry.name));
            std::fs::write(&path, ppm)?;
            println!("{}", path.display());
        }
        eprintln!(
            "qimage-loop: sequence of {} requests in {:.3}s",
            spec.requests.len(),
            sequence.elapsed().as_secs_f64()
        );
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
    {
        native::run()
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "mlx")))]
    {
        Err("qwen_image_loop requires Apple Silicon macOS and the mlx feature".into())
    }
}

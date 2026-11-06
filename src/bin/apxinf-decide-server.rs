//! Resident d3-lite decision process for the Mario demo (`APXINF_DECIDE_CMD`).
//!
//! ```text
//! apxinf-decide-server --model <checkpoint> [--model-name NAME] [--device metal:N]
//! ```
//!
//! One checkpoint, one `{"type":"ready",…}` line on stdout, then one JSONL
//! request → one JSONL response per stdin line:
//!
//! ```text
//! {"id":1,"state":"…","questions":{"enemy":{"type":"noul","instructions":"…"}},
//!  "images":["<base64 png>"]}
//! {"id":1,"answers":{"enemy":{"type":"noul","noul":0.42}},
//!  "latency_ms":184.2,"stages":{…}}
//! ```
//!
//! This is the Clef `apxinf-decide-server` protocol, so the demo's
//! `clef_local.ClefLocalWorker` drives it unchanged: no new bridge, and the
//! Clef bridge's behaviour is untouched. A malformed request answers
//! `{"id":…,"error":"…"}` and the process keeps serving; only a load failure,
//! exhausted stdin or a closed stdout ends it.
//!
//! **Image contract.** d3-lite writes one contiguous image-placeholder run
//! (P1b), so one request carries at most one image. The demo's separate
//! protocol (`MARIO_CLEF_PROTOCOL=separate`, the Clef reference's own startup
//! parameter) is one crop + one Noul question per request; `joint` would send
//! three images and is rejected by name.
//!
//! The request clock covers base64 + PNG decode, the checkpoint's
//! resize/normalize/patchify, prompt render + tokenize and the model call —
//! the same boundary the Clef server and `d3_lite_bench` use.
//!
//! The stage split (`stages`) is the runner's host wall-clock attribution:
//! `host_prep` (decode + preprocess + prompt encode), then `layout`, `vision`,
//! `forward` (text layers + FP32 readout) and `sync` from the model profile.

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod live {
    use std::io::{BufRead, Write};
    use std::path::PathBuf;
    use std::time::Instant;

    use apxinf_core::{Device, Tensor};
    use apxinf_model::metal_d3_lite::{preprocess, D3LiteFormat};
    use apxinf_model::{
        AutoModel, DecisionInput, DecisionOutput, DecisionQuestion, ImageInput, LoadOptions,
        LoadedModel,
    };
    use base64::decode as base64_decode;
    use serde_json::{json, Map, Value};

    type Fallible<T> = Result<T, String>;

    pub fn main() {
        match load() {
            Ok(mut server) => serve(&mut server),
            Err(error) => {
                // The demo client reads this as a failed start-up and shows
                // the text.
                let _ = writeln!(
                    std::io::stdout(),
                    "{}",
                    json!({"type": "error", "error": error})
                );
                let _ = std::io::stdout().flush();
                std::process::exit(1);
            }
        }
    }

    #[derive(Debug)]
    struct Options {
        model: PathBuf,
        model_name: Option<String>,
        device: Device,
    }

    fn parse_options() -> Fallible<Options> {
        let mut model = None;
        let mut model_name = None;
        let mut device = Device::Metal(0);
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" | "--snapshot" => {
                    model = Some(PathBuf::from(args.next().ok_or("--model needs a path")?))
                }
                "--model-name" => {
                    model_name = Some(args.next().ok_or("--model-name needs a value")?)
                }
                "--device" => {
                    let value = args.next().ok_or("--device needs a value")?;
                    device = parse_device(&value)?
                }
                "--help" | "-h" => {
                    return Err("usage: apxinf-decide-server --model <checkpoint> \
                                 [--model-name NAME] [--device metal:N]"
                        .into())
                }
                other if model.is_none() && !other.starts_with('-') => {
                    model = Some(PathBuf::from(other))
                }
                other => return Err(format!("unexpected argument {other}")),
            }
        }
        Ok(Options {
            model: model.ok_or("--model <checkpoint> is required")?,
            model_name,
            device,
        })
    }

    fn parse_device(value: &str) -> Fallible<Device> {
        let (kind, index) = value
            .split_once(':')
            .map_or((value, Some(0usize)), |(kind, index)| {
                (kind, index.parse::<usize>().ok())
            });
        match (kind, index) {
            ("metal", Some(index)) => Ok(Device::Metal(index)),
            ("metal", None) => Err("--device metal:N needs a numeric index".into()),
            _ => Err(format!(
                "unsupported device {value:?}; this server runs metal:N"
            )),
        }
    }

    /// Load the resident model and the prompt contract, verify the package's
    /// code→token mapping against the tokenizer, print the one `ready` line.
    struct Server {
        model: LoadedModel,
        format: D3LiteFormat,
        model_name: String,
    }

    fn load() -> Fallible<Server> {
        let options = parse_options()?;
        let started = Instant::now();
        let format = D3LiteFormat::from_checkpoint(&options.model)
            .map_err(|error| format!("format: {error}"))?;
        // Prove the decision_config codes/token IDs are the tokenizer's own
        // result (`d3_format.answer_codes`), not a transcription, before the
        // model is loaded.
        let (probe_codes, probe_ids) = format
            .answer_codes()
            .map_err(|error| format!("answer-code probe: {error}"))?;
        if probe_codes != format.package_codes() || probe_ids != format.package_token_ids() {
            return Err(
                "d3_format.answer_codes disagrees with decision_config.json codes/token_ids".into(),
            );
        }
        let mut load_options = LoadOptions::default();
        load_options.model_name = options.model_name.clone();
        let model = AutoModel::load_model(options.device, &options.model, &load_options)
            .map_err(|error| format!("load: {error}"))?;
        let load_ms = started.elapsed().as_secs_f64() * 1e3;
        let model_name = load_options
            .model_name
            .or_else(|| AutoModel::detect_model_name(&options.model).ok())
            .unwrap_or_else(|| "metal_d3".into());
        // The Metal port always executes the compiled whole-request callable
        // over BF16 weights with the FP32 readout (`admit` accepts no other
        // precision), so `bf16-compiled` is the effective variant.
        let variant = "bf16-compiled";
        writeln!(
            std::io::stdout(),
            "{}",
            json!({
                "type": "ready",
                "model": model_name,
                "variant": variant,
                "load_ms": load_ms,
                "codes_verified": true,
            })
        )
        .map_err(|error| format!("write ready: {error}"))?;
        std::io::stdout()
            .flush()
            .map_err(|error| format!("flush ready: {error}"))?;
        Ok(Server {
            model,
            format,
            model_name,
        })
    }

    fn serve(server: &mut Server) {
        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(line) => line,
                Err(error) => {
                    eprintln!("stdin: {error}");
                    break;
                }
            };
            if line.trim().is_empty() {
                continue;
            }
            let response = handle(&mut server.model, &server.format, &server.model_name, &line);
            if writeln!(stdout, "{response}").is_err() || stdout.flush().is_err() {
                break;
            }
        }
    }

    /// One request line in, one response line out; never fails the process.
    fn handle(
        model: &mut LoadedModel,
        format: &D3LiteFormat,
        model_name: &str,
        line: &str,
    ) -> Value {
        let request: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(error) => {
                return json!({"id": Value::Null, "error": format!("request is not JSON: {error}")})
            }
        };
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        match answer(model, format, &request) {
            Ok((answers, latency_ms, stages)) => json!({
                "id": id,
                "answers": answers,
                "latency_ms": latency_ms,
                "stages": stages,
                "model": format_package(format, model_name),
            }),
            Err(error) => json!({"id": id, "error": error}),
        }
    }

    fn format_package(format: &D3LiteFormat, model_name: &str) -> Value {
        json!({
            "name": model_name,
            "format_id": apxinf_model::metal_d3_lite::FORMAT_ID,
            "codes": format.package_codes().len(),
        })
    }

    /// One request: every question is one prompt; with an image, every
    /// question sees the request's single image (the separate protocol asks
    /// one question per image).
    fn answer(
        model: &mut LoadedModel,
        format: &D3LiteFormat,
        request: &Value,
    ) -> Fallible<(Value, f64, Value)> {
        let started = Instant::now();
        let state = request.get("state").ok_or("request needs state")?;
        let questions = request
            .get("questions")
            .and_then(Value::as_object)
            .ok_or("request needs a questions object")?;
        if questions.is_empty() {
            return Err("request needs at least one question".into());
        }
        let images = request.get("images").and_then(Value::as_array);
        if let Some(images) = images {
            if images.len() > 1 {
                return Err("d3-lite Metal writes one image run per request; use \
                             MARIO_CLEF_PROTOCOL=separate (one crop + one question)"
                    .into());
            }
        }
        let image = images
            .and_then(|images| images.first())
            .map(|value| {
                value
                    .as_str()
                    .ok_or_else(|| "images must be base64 PNG strings".to_string())
            })
            .transpose()?
            .map(decode_png)
            .transpose()?;

        let mut answers = Map::with_capacity(questions.len());
        let mut host_prep_ms = 0.0f64;
        let mut layout_ms = 0.0f64;
        let mut vision_ms = 0.0f64;
        let mut forward_ms = 0.0f64;
        let mut sync_ms = 0.0f64;
        for (id, raw_question) in questions {
            let question = decision_question(id, raw_question)?;
            let prepared_at = Instant::now();
            let prepared = PreparedQuestion::build(format, state, &question, image.as_ref())?;
            host_prep_ms += prepared_at.elapsed().as_secs_f64() * 1e3;

            let output = model
                .decide(prepared.input())
                .map_err(|error| error.to_string())?;
            let profile = model
                .decision_mut()
                .map_err(|error| error.to_string())?
                .last_stage_profile()
                .ok_or_else(|| "d3-lite runner did not record a stage profile".to_string())?;
            layout_ms += profile.layout_ms;
            vision_ms += profile.vision_ms;
            forward_ms += profile.forward_ms;
            sync_ms += profile.sync_ms;
            let probabilities = prepared.option_probabilities(&output);
            let parsed = format
                .answer(prepared.normalized(), &probabilities)
                .map_err(|error| error.to_string())?;
            answers.insert(id.clone(), parsed);
        }
        let latency_ms = started.elapsed().as_secs_f64() * 1e3;
        let stages = json!({
            "host_prep_ms": host_prep_ms,
            "layout_ms": layout_ms,
            "vision_ms": vision_ms,
            "forward_ms": forward_ms,
            "sync_ms": sync_ms,
            "questions": questions.len(),
        });
        Ok((Value::Object(answers), latency_ms, stages))
    }

    /// A request question in the model-neutral schema shape.
    fn decision_question(id: &str, value: &Value) -> Fallible<DecisionQuestion> {
        let object = value
            .as_object()
            .ok_or_else(|| format!("question {id} must be an object"))?;
        let kind = object
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("question {id} needs a `type`"))?
            .to_string();
        for key in object.keys() {
            if !matches!(key.as_str(), "type" | "instructions" | "criteria") {
                return Err(format!("question {id} has unknown field `{key}`"));
            }
        }
        Ok(DecisionQuestion {
            id: id.to_string(),
            kind,
            instructions: object.get("instructions").cloned(),
            criteria: object.get("criteria").cloned(),
        })
    }

    /// base64 → RGB8 + dimensions. The demo sends PNG; the checkpoint's
    /// preprocessor is bit-exact for PNG (P1b).
    fn decode_png(text: &str) -> Fallible<(Vec<u8>, usize, usize)> {
        if text.len() > 24 * 1024 * 1024 {
            return Err("image base64 exceeds 24 MiB".into());
        }
        let bytes =
            base64_decode(text).map_err(|error| format!("image is not valid base64: {error}"))?;
        let decoded =
            image::load_from_memory(&bytes).map_err(|error| format!("image decode: {error}"))?;
        let rgb = decoded.to_rgb8();
        let (width, height) = image::GenericImageView::dimensions(&rgb);
        Ok((rgb.into_raw(), width as usize, height as usize))
    }

    /// Everything one question needs to reach the model, owned so the
    /// borrowed [`DecisionInput`] stays valid across `model.decide`.
    enum PreparedQuestion {
        Text {
            encoded: apxinf_model::metal_d3_lite::EncodedRequest,
        },
        Image {
            encoded: apxinf_model::metal_d3_lite::EncodedImageRequest,
            pixels: Tensor,
            grids: [[u32; 3]; 1],
        },
    }

    impl PreparedQuestion {
        fn build(
            format: &D3LiteFormat,
            state: &Value,
            question: &DecisionQuestion,
            image: Option<&(Vec<u8>, usize, usize)>,
        ) -> Fallible<Self> {
            if let Some((rgb, width, height)) = image {
                // The checkpoint's resize/normalize/patchify contract runs on
                // the host; `PreparedImage` owns the pixel tensor and grid.
                let prepared = preprocess(
                    &format.config().preprocessor,
                    &format.config().vision,
                    rgb,
                    *width,
                    *height,
                )
                .map_err(|error| format!("preprocess: {error}"))?;
                let rows = prepared.grid_thw[1] as usize * prepared.grid_thw[2] as usize;
                let patch_dim = format.patch_dim();
                if prepared.pixel_values.len() != rows * patch_dim {
                    return Err(format!(
                        "preprocess produced {} values, expected {rows} x {patch_dim}",
                        prepared.pixel_values.len()
                    ));
                }
                let pixels = Tensor::from_f32(vec![rows, patch_dim], &prepared.pixel_values)
                    .map_err(|error| format!("pixel tensor: {error}"))?;
                let grids = [prepared.grid_thw];
                let encoded = format
                    .encode_image(state, question, &prepared)
                    .map_err(|error| error.to_string())?;
                Ok(Self::Image {
                    encoded,
                    pixels,
                    grids,
                })
            } else {
                let encoded = format
                    .encode_text(state, question)
                    .map_err(|error| error.to_string())?;
                Ok(Self::Text { encoded })
            }
        }

        fn input(&self) -> DecisionInput<'_> {
            match self {
                Self::Text { encoded } => {
                    DecisionInput::text(&encoded.token_ids, encoded.candidate_count)
                }
                Self::Image {
                    encoded,
                    pixels,
                    grids,
                    ..
                } => DecisionInput::with_vision(
                    &encoded.token_ids,
                    ImageInput::new(pixels, grids),
                    encoded.candidate_count,
                ),
            }
        }

        fn normalized(&self) -> &apxinf_model::metal_d3_lite::NormalizedQuestion {
            match self {
                Self::Text { encoded } => &encoded.normalized,
                Self::Image { encoded, .. } => &encoded.normalized,
            }
        }

        fn option_probabilities(&self, output: &DecisionOutput) -> Vec<f32> {
            match self {
                Self::Text { encoded } => {
                    apxinf_model::metal_d3_lite::D3LiteFormat::option_probabilities(encoded, output)
                }
                Self::Image { encoded, .. } => {
                    apxinf_model::metal_d3_lite::D3LiteFormat::image_option_probabilities(
                        encoded, output,
                    )
                }
            }
        }
    }
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn main() {
    live::main();
}

/// The resident server is Apple-Silicon + MLX only, matching the
/// `metal_d3_lite` module's gate. Other targets still compile this binary so
/// `cargo check --features mlx` (the Linux feature-isolation CI job) stays
/// green; the process refuses to serve.
#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!(
        "apxinf-decide-server requires --features mlx on Apple Silicon \
         (this build: feature={}, target={})",
        if cfg!(feature = "mlx") {
            "mlx"
        } else {
            "no-mlx"
        },
        std::env::consts::OS,
    );
    std::process::exit(1);
}

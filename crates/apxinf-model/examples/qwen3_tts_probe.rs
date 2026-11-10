//! End-to-end Qwen3-TTS measurement: time to first audio and real-time factor.
use apxinf_core::Device;
use apxinf_model::{AutoModel, LoadedModel, SpeechRequest, SpeechRuntime};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err("usage: qwen3_tts_probe CHECKPOINT TEXT [VOICE] [LANGUAGE]".into());
    }
    let checkpoint = Path::new(&args[0]);
    let text = args[1].clone();
    let voice = args.get(2).map(|s| s.as_str()).unwrap_or("vivian");
    let language = args.get(3).map(|s| s.as_str()).unwrap_or("auto");

    let loaded = AutoModel::load_model(
        Device::Metal(0),
        checkpoint,
        &Default::default(),
    )?;
    let mut runtime = match loaded {
        LoadedModel::Speech(runtime) => runtime,
        _ => return Err("checkpoint did not load as a speech runtime".into()),
    };
    let request = SpeechRequest {
        text: &text,
        voice,
        language,
        instruction: "",
        seed: 7,
    };
    let started = Instant::now();
    let audio = runtime.synthesize(&request)?;
    let elapsed = started.elapsed().as_secs_f64();
    let seconds = audio.samples.len() as f64 / (audio.sample_rate as f64 * audio.channels as f64);
    println!(
        "{{\"kind\":\"tts\",\"sample_rate\":{},\"channels\":{},\"samples\":{},\"audio_seconds\":{:.3},\
\"wall_seconds\":{:.3},\"rtf\":{:.3},\"complete\":{}}}",
        audio.sample_rate,
        audio.channels,
        audio.samples.len(),
        seconds,
        elapsed,
        elapsed / seconds,
        audio.complete
    );
    Ok(())
}

//! Pure-CPU stage probe for the GPU/ANE-independent front of a Clef request.
//!
//! `decide_schema` does decode → resize → patchify → tokenize before it ever
//! touches a Metal array, but `ClefRunner::load` rejects `Device::Cpu`, so
//! that front cannot be timed through the model API. [`preprocess_and_encode`]
//! runs exactly those two steps (same functions, same inputs) and times each
//! sub-stage, without loading weights, creating a stream or taking the device
//! lease. Nothing here is a model measurement.

use super::{encode, preprocess};
use crate::decision::{DecisionImage, DecisionSchema};
use apxinf_core::Result;
use apxinf_tokenizer::Tokenizer;
use std::path::Path;
use std::time::Instant;

/// Milliseconds to print one stage on stderr as
/// `APXINF_HOST_STAGE stage=<name> ms=<value>`.
fn stamp(stage: &str, started: Instant) {
    eprintln!(
        "APXINF_HOST_STAGE stage={stage} ms={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );
}

/// Decode, resize, patchify and tokenize one image request, timing every
/// sub-stage. Mirrors the front of `ClefRunner::decide_schema`
/// (`mod.rs:179-192`) with the same functions and arguments:
/// `preprocess::prepare` on `DecisionImage::Path`, then
/// `encode::encode_with_images` with `config.json:image_token_id`. Returns
/// the prompt's token count so a caller can cross-check the 299 / 303 / 323
/// token classes of the 42-case reference.
///
/// No checkpoint weights, no `Stream`, no Metal and no locks are touched.
pub fn preprocess_and_encode(
    schema: &DecisionSchema,
    image: &Path,
    tokenizer_path: &Path,
    image_token_id: u32,
) -> Result<usize> {
    let started = Instant::now();
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    stamp("tokenizer_load", started);

    // `preprocess::prepare` on one path is decode → smart_resize →
    // bicubic → patchify. Time it whole (that is exactly what the request
    // runs), then re-run the decode/resize sub-stages on the same bytes so
    // their individual shares are visible without changing `prepare`.
    let images = [DecisionImage::Path(image)];
    let started = Instant::now();
    let prepared = preprocess::prepare(&images)?;
    stamp("prepare_total", started);

    let bytes = std::fs::read(image)?;
    let (width, height, rgb) = preprocess::decode_png(&bytes)?;
    let started = Instant::now();
    let (resized_height, resized_width) = preprocess::smart_resize(height, width)?;
    stamp("smart_resize", started);
    let started = Instant::now();
    let resized = preprocess::resize_bicubic(&rgb, width, height, resized_width, resized_height);
    stamp("bicubic", started);
    let _ = resized;
    let _ = resized;

    let started = Instant::now();
    let encoded = encode::encode_with_images(
        &tokenizer,
        schema,
        &prepared.grids,
        image_token_id,
    )?;
    stamp("encode", started);
    Ok(encoded.input_ids.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const SUITE: &str = "/Users/haiyan/Documents/Infinity/Agent4Kernel/mario-showcase-20261103/suite/separate/separate-proto14.json";
    const CHECKPOINT: &str = "/Users/haiyan/Documents/Infinity/Agent4Kernel/models/clef-flash-4bit-b5adbe2";

    fn schema_from(record: &Value) -> Result<DecisionSchema> {
        let body = serde_json::json!({
            "state": record.get("state").cloned().unwrap_or(Value::Null),
            "questions": record.get("questions").cloned().unwrap_or_else(|| Value::Object(Default::default())),
        });
        DecisionSchema::from_json(&body.to_string())
    }

    #[test]
    fn all_42_single_image_fronts_encode_without_a_device() -> Result<()> {
        let suite: Value =
            serde_json::from_str(&std::fs::read_to_string(SUITE).map_err(|e| {
                apxinf_core::Error::Other(format!("read suite: {e}"))
            })?)
            .map_err(|e| apxinf_core::Error::Other(format!("parse suite: {e}")))?;
        let cases = suite
            .get("cases")
            .and_then(Value::as_array)
            .ok_or_else(|| apxinf_core::Error::Other("suite has no cases".into()))?;
        assert_eq!(cases.len(), 42, "the parity suite is 42 single-image cases");
        let config: Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{CHECKPOINT}/config.json")).map_err(|e| {
                apxinf_core::Error::Other(format!("read config: {e}"))
            })?,
        )
        .map_err(|e| apxinf_core::Error::Other(format!("parse config: {e}")))?;
        let image_token_id = config["image_token_id"]
            .as_u64()
            .ok_or_else(|| apxinf_core::Error::Other("config has no image_token_id".into()))?
            as u32;
        let tokenizer_path = Path::new(CHECKPOINT).join("tokenizer.json");
        let mut counts = Vec::with_capacity(cases.len());
        for case in cases {
            let record = case
                .get("record")
                .ok_or_else(|| apxinf_core::Error::Other("case has no record".into()))?;
            let schema = schema_from(record)?;
            let image = record["images"]
                .as_array()
                .and_then(|images| images.first())
                .and_then(Value::as_str)
                .ok_or_else(|| apxinf_core::Error::Other("case has no image path".into()))?;
            let count = preprocess_and_encode(
                &schema,
                Path::new(image),
                &tokenizer_path,
                image_token_id,
            )?;
            counts.push(count);
        }
        counts.sort_unstable();
        // The 42-case reference is 299 / 303 / 323 tokens (PARITY-EVIDENCE §2).
        assert_eq!(counts[0], 299, "smallest prompt must match the reference");
        assert_eq!(*counts.last().unwrap(), 323, "largest prompt must match the reference");
        Ok(())
    }
}

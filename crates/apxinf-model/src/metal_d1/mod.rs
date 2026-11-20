//! LiquidAI d1-2B native MLX/Metal decision model (P1a: text path).
//!
//! `metal_d1` is the Metal port of `LiquidAI/d1-2B` @ `a6bce774`: a 1.92B
//! decision model whose text backbone is field-for-field LFM3-1.7B (copied
//! from `metal_lfm3` with provenance, never imported), plus the D1 System-One
//! readout (set attention, slot logits, pointer spans, calibration). The
//! family is **non-generative**: one prefill pass per packed sequence, typed
//! answers out — it is not an `LlmTrait` model.
//!
//! P1a scope: text-only requests (no vision tower, no batching), validated
//! against the frozen P0 fixtures. Registration through `AutoModel` and the
//! shared decision protocol lands with the confirmed `decision.rs` contract
//! (issue #123 §6), not here.

pub mod calibration;
pub mod config;
mod model;
pub mod parsing;
pub mod prompt;
pub mod questions;
pub mod readout;
pub mod runner;
mod weights;

pub use config::D1Config;
pub use model::{D1Model, SetAttention};

use apxinf_core::{Backend, Error, Result};
use std::path::Path;
use std::sync::Arc;

/// Load the family directly (P1a harness path; `AutoModel` wiring is P2).
pub fn load(path: &Path, backend: Arc<dyn Backend>) -> Result<D1Model> {
    let json = std::fs::read_to_string(path.join("config.json"))
        .map_err(|e| Error::Other(format!("cannot read d1 config: {e}")))?;
    let config = D1Config::from_json(&json)?;
    let (tensors, _) = apxinf_loader::safetensors::load_native_path(path).map_err(Error::Other)?;
    // P1a: the text backbone only. Vision tower and projector tensors live in
    // the same file and are admitted (unused) until P1b.
    let text: std::collections::HashMap<_, _> = tensors
        .into_iter()
        .filter(|(name, _)| name.starts_with("model.language_model."))
        .collect();
    if text.is_empty() {
        return Err(Error::Other(
            "d1 checkpoint carries no model.language_model.* tensors".into(),
        ));
    }
    D1Model::new(config, text, backend)
}

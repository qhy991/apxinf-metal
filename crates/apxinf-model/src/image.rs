//! Model-neutral text-to-image API.

use apxinf_core::Result;

/// One complete text-to-image request. Families document the sizes and step
/// counts they support and reject others before any device work.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageRequest<'a> {
    pub prompt: &'a str,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    pub steps: usize,
    /// TeaCache-style step reuse. `None` uses the family default, `Some(0.0)`
    /// disables reuse (rollback), and `Some(r)` with `0 < r < 1` selects the
    /// fraction of denoise steps whose model output is reused. Families reject
    /// non-finite or out-of-range values before any device work.
    pub step_cache_ratio: Option<f32>,
}

/// A generated image owned by the caller: row-major RGB8, `width * height * 3`
/// bytes. Later requests never overwrite it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedImage {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

/// Text-to-image runtime. Implementations own their weights and preparation;
/// request state (prompt embeddings, latents, noise) lives for one request.
pub trait ImageRuntime {
    fn generate(&mut self, request: &ImageRequest<'_>) -> Result<GeneratedImage>;
}

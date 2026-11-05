//! Model weight loading from SafeTensors and GGUF formats.

pub mod config;
#[cfg(feature = "gguf")]
pub mod gguf;
#[cfg(feature = "safetensors")]
pub mod safetensors;

pub use config::ModelConfig;

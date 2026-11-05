use std::env;

fn main() {
    println!("cargo:rerun-if-changed=native/signpost_shim.c");
    if env::var_os("CARGO_FEATURE_MLX").is_none() {
        return;
    }
    // The `metal_clef` module is only compiled for Apple Silicon; the same
    // guard keeps the Linux/CUDA feature-isolation build SDK-free.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64")
    {
        return;
    }
    cc::Build::new()
        .file("native/signpost_shim.c")
        .flag("-fvisibility=hidden")
        .compile("apxinf_model_signpost");
}

use std::path::Path;

fn read_shader(name: &str) -> String {
    let path = format!("src/{name}.metal");
    println!("cargo:rerun-if-changed={path}");
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"))
}

fn embed_shader(output_dir: &Path, name: &str, symbol: &str, source: &str) {
    const DELIMITER: &str = "APX_METAL";
    assert!(
        !source.contains(&format!("){DELIMITER}\"")),
        "Metal shader contains the C++ raw-string delimiter"
    );
    std::fs::write(
        output_dir.join(format!("{name}_source.inc")),
        format!("constexpr const char *{symbol} = R\"{DELIMITER}({source}){DELIMITER}\";\n"),
    )
    .expect("write the embedded Metal shader include");
}

fn compile_bridge(output_dir: &Path, name: &str, experiments: bool) {
    let path = format!("src/{name}_bridge.mm");
    println!("cargo:rerun-if-changed={path}");
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file(path)
        .include(output_dir)
        .flag("-std=c++17")
        .flag("-fobjc-arc")
        .flag("-fblocks");
    if experiments {
        build.define("APXINF_METAL_EXPERIMENTS", None);
    }
    build.compile(&format!("apxinf_{name}_bridge"));
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || std::env::var_os("CARGO_FEATURE_W8_HEAD_MLP").is_none()
    {
        return;
    }

    let output_dir =
        std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo provides OUT_DIR"));
    let experiments = std::env::var_os("CARGO_FEATURE_EXPERIMENTS").is_some();
    let head = read_shader("metal_w8");
    let mlp = read_shader("metal_w8_mlp");
    embed_shader(&output_dir, "metal_w8", "kMetalSource", &head);
    embed_shader(&output_dir, "metal_w8_mlp", "kMetalMlpSource", &mlp);

    if experiments {
        let matvec = read_shader("metal_w8_matvec");
        let gdn = read_shader("metal_w8_gdn");
        let linear = read_shader("metal_w8_linear_layer");
        let gdn_out_g32 = read_shader("metal_w8_gdn_out_g32");
        let full_attention = read_shader("metal_full_attention_decode_v1");
        embed_shader(
            &output_dir,
            "metal_w8_matvec",
            "kMetalMatVecSource",
            &matvec,
        );
        embed_shader(&output_dir, "metal_w8_gdn", "kMetalGdnSource", &gdn);
        embed_shader(
            &output_dir,
            "metal_w8_linear_layer",
            "kMetalLinearLayerSource",
            &format!("{gdn}\n{mlp}\n{linear}\n{gdn_out_g32}"),
        );
        embed_shader(
            &output_dir,
            "metal_w8_tail_mlp_head_v1",
            "kMetalTailMlpHeadSourceV1",
            &format!("{mlp}\n{linear}\n{head}"),
        );
        embed_shader(
            &output_dir,
            "metal_full_attention_decode_v1",
            "kMetalFullAttentionDecodeSourceV1",
            &full_attention,
        );
    }

    compile_bridge(&output_dir, "metal_w8", experiments);
    compile_bridge(&output_dir, "metal_w8_mlp", experiments);
    if experiments {
        for bridge in [
            "metal_w8_gdn",
            "metal_w8_linear_layer",
            "metal_w8_linear_layer_stack3",
            "metal_w8_mlp_stack3_boundary_v1",
            "metal_w8_tail_mlp_head_v1",
            "metal_gdn_recurrent_count18_profile_v1",
            "metal_gdn_core_fused_count18_profile_v1",
            "metal_full_attention_decode_v1",
        ] {
            compile_bridge(&output_dir, bridge, experiments);
        }
    }

    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
}

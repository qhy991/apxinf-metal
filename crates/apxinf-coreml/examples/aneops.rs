//! aneops: which MIL op classes the compiler places on the Neural Engine, and
//! at what shape.
//!
//! Each case in a fixed table is authored with `ProgramBuilder`, compiled for
//! `CpuAndNeuralEngine`, and loaded. The compiler's own `MLComputePlan` then
//! gives every operation's preferred device, and the case is timed with its
//! declared inputs (median of 20 predictions after 5 warmups). A case the
//! builder cannot author, or the compiler refuses, is a result and is printed
//! like any other.
//!
//!   cargo run -q -p apxinf-coreml --release --features native --example aneops
use apxinf_core::{Error, Result};
use apxinf_coreml::{ComputeDeviceKind, ComputeUnits, Operand, Package, ProgramBuilder, ProgramCache, Var};
use half::f16;
use std::time::{Duration, Instant};

const WARMUP: usize = 5;
const TIMED: usize = 20;

/// An authored case: the package and the values it is fed and read through.
struct Program {
    package: Package,
    inputs: Vec<Var>,
    outputs: Vec<Var>,
    /// Where the authored program differs from the case as specified.
    note: &'static str,
}

struct Case {
    name: &'static str,
    shape: String,
    author: Box<dyn Fn() -> Result<Program>>,
    /// Cases that must place entirely on the engine before this one is worth building.
    requires: &'static [&'static str],
}

enum Outcome {
    /// A required case did not place entirely on the engine.
    Blocked(Vec<String>),
    /// The builder refused to author the case.
    Unsupported(String),
    /// Compile, load or prediction failed.
    Refused(String),
    Measured(Measurement),
}

struct Measurement {
    on_ane: Vec<String>,
    off_ane: Vec<String>,
    no_device: usize,
    ms: f64,
    ms_min: f64,
    ms_max: f64,
    finite: bool,
    nonzero: bool,
    note: &'static str,
}

fn values(n: usize, seed: usize) -> Vec<f16> {
    (0..n)
        .map(|i| f16::from_f32((((i + seed) * 40503) % 200) as f32 / 100.0 - 1.0))
        .collect()
}

fn weights(n: usize) -> Vec<f16> {
    (0..n)
        .map(|i| f16::from_f32((((i * 2654435761) % 2000) as f32 / 1000.0 - 1.0) * 0.02))
        .collect()
}

fn finish(b: ProgramBuilder, inputs: Vec<Var>, outputs: &[&Var], note: &'static str) -> Result<Program> {
    Ok(Program {
        package: b.finish(outputs)?,
        inputs,
        outputs: outputs.iter().map(|&v| v.clone()).collect(),
        note,
    })
}

fn gemm(tile: usize) -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 2048, 1, tile]);
    let y = b.conv_fp16(&x, &weights(6144 * 2048), &[6144, 2048, 1, 1])?;
    finish(b, vec![x], &[&y], "")
}

fn conv2d_3x3() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 128, 64, 64]);
    let y = b.conv_fp16(&x, &weights(128 * 128 * 9), &[128, 128, 3, 3])?;
    finish(
        b,
        vec![x],
        &[&y],
        "pad_type valid, output [1,128,62,62]: conv_fp16 emits only valid padding, so same is not authorable",
    )
}

fn rmsnorm() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 2048, 1, 1408]);
    let gamma = b.input("gamma", &[1, 2048, 1, 1]);
    let sq = b.mul(&x, Operand::Var(&x), None)?;
    let sum = b.reduce_sum(&sq, &[1], true)?;
    let r = b.rsqrt(&sum);
    let normed = b.mul(&x, Operand::Var(&r), None)?;
    let y = b.mul(&normed, Operand::Var(&gamma), None)?;
    finish(
        b,
        vec![x, gamma],
        &[&y],
        "gamma is an input [1,2048,1,1], not a constant: the builder has no public fp16 constant, and [2048,1,1,1] would broadcast x to [2048,2048,1,1408]; no eps, 1/C folds into gamma",
    )
}

/// `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`.
fn tanh_gelu(b: &mut ProgramBuilder, x: &Var) -> Result<Var> {
    let s = |v: f32| Operand::Scalar(f16::from_f32(v));
    let x2 = b.mul(x, Operand::Var(x), None)?;
    let x3 = b.mul(&x2, Operand::Var(x), None)?;
    let cubic = b.mul(&x3, s(0.044715), None)?;
    let inner = b.add(x, Operand::Var(&cubic), None)?;
    let scaled = b.mul(&inner, s((2.0 / std::f32::consts::PI).sqrt()), None)?;
    let t = b.tanh(&scaled);
    let one_plus = b.add(&t, s(1.0), None)?;
    let half = b.mul(x, s(0.5), None)?;
    b.mul(&half, Operand::Var(&one_plus), None)
}

fn gelu() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 6144, 1, 1408]);
    let y = tanh_gelu(&mut b, &x)?;
    finish(b, vec![x], &[&y], "")
}

fn gemm_plus_gelu() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 2048, 1, 1408]);
    let h = b.conv_fp16(&x, &weights(6144 * 2048), &[6144, 2048, 1, 1])?;
    let y = tanh_gelu(&mut b, &h)?;
    finish(b, vec![x], &[&y], "")
}

fn softmax_op() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 1, 128, 1408]);
    let y = b.softmax(&x, 3)?;
    finish(b, vec![x], &[&y], "")
}

fn conv_dynamic_weight() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 256, 1, 1408]);
    let w = b.input("w", &[128, 256, 1, 1]);
    let y = b.conv(&x, &w)?;
    finish(b, vec![x, w], &[&y], "")
}

const HEADS: usize = 8;
const SEQ: usize = 1408;
const HEAD_DIM: usize = 128;

/// Per head: `S = conv(Q_h, K_h)` is `[1, keys, 1, queries]`, softmax over the
/// key (channel) axis, then `O_h = conv(P, V_h^T)` is `[1, head_dim, 1, queries]`.
fn attention_core_conv() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let q = b.input("q", &[1, HEADS * HEAD_DIM, 1, SEQ]);
    let k = b.input("k", &[HEADS * SEQ, HEAD_DIM, 1, 1]);
    let v = b.input("v", &[HEADS * HEAD_DIM, SEQ, 1, 1]);
    let qs = b.split(&q, &[HEAD_DIM; HEADS], 1)?;
    let ks = b.split(&k, &[SEQ; HEADS], 0)?;
    let vs = b.split(&v, &[HEAD_DIM; HEADS], 0)?;
    let mut heads = Vec::with_capacity(HEADS);
    for h in 0..HEADS {
        let scores = b.conv(&qs[h], &ks[h])?;
        let p = b.softmax(&scores, 1)?;
        heads.push(b.conv(&p, &vs[h])?);
    }
    let y = b.concat(&heads.iter().collect::<Vec<_>>(), 1, false)?;
    finish(
        b,
        vec![q, k, v],
        &[&y],
        "one conv pair per head: the builder's conv has groups=1, so 8 heads cannot share one grouped conv; q/k/v arrive in conv layout and are split per head, output [1,1024,1,1408]; no 1/sqrt(d) scale",
    )
}

fn reshape_matmul() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 8, 1408, 256]);
    let value = b.input("value", &[8, 256, 1408]);
    let r = b.reshape(&x, &[8, 1408, 256])?;
    let y = b.matmul(&r, &value)?;
    finish(b, vec![x, value], &[&y], "")
}

fn softmax_last_axis(b: &mut ProgramBuilder, x: &Var) -> Result<Var> {
    let e = b.exp(x);
    let sum = b.reduce_sum(&e, &[x.shape().len() as i32 - 1], true)?;
    b.real_div(&e, Operand::Var(&sum), None)
}

fn softmax() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 1, 128, 1408]);
    let y = softmax_last_axis(&mut b, &x)?;
    finish(b, vec![x], &[&y], "")
}

fn attention_core() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let scores = b.input("scores", &[1, 8, 1408, 256]);
    let value = b.input("value", &[1, 8, 256, 1408]);
    let out_value = b.input("out_value", &[1, 8, 1408, 256]);
    let s = b.matmul(&scores, &value)?;
    let p = softmax_last_axis(&mut b, &s)?;
    let y = b.matmul(&p, &out_value)?;
    finish(b, vec![scores, value, out_value], &[&y], "")
}

fn transpose_head() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 8, 1408, 256]);
    let y = b.transpose(&x, &[0, 1, 3, 2])?;
    finish(b, vec![x], &[&y], "")
}

fn concat_heads() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let heads: Vec<Var> = (0..4)
        .map(|i| b.input(&format!("h{i}"), &[1, 8, 1408, 64]))
        .collect();
    let y = b.concat(&heads.iter().collect::<Vec<_>>(), 3, false)?;
    finish(b, heads, &[&y], "")
}

fn cases() -> Vec<Case> {
    let case = |name: &'static str, shape: &str, author: fn() -> Result<Program>| Case {
        name,
        shape: shape.to_owned(),
        author: Box::new(author),
        requires: &[],
    };
    let mut cases = vec![Case {
        name: "gemm_baseline",
        shape: "x[1,2048,1,1408] w[6144,2048,1,1]".into(),
        author: Box::new(|| gemm(1408)),
        requires: &[],
    }];
    for tile in [1, 4, 8, 16, 32, 64, 128, 256, 512] {
        cases.push(Case {
            name: "gemm_small_m",
            shape: format!("x[1,2048,1,{tile}] w[6144,2048,1,1]"),
            author: Box::new(move || gemm(tile)),
            requires: &[],
        });
    }
    cases.extend([
        case("conv2d_3x3", "x[1,128,64,64] w[128,128,3,3]", conv2d_3x3),
        case("rmsnorm", "x[1,2048,1,1408]", rmsnorm),
        case("gelu", "x[1,6144,1,1408]", gelu),
        case("softmax_last_axis", "x[1,1,128,1408]", softmax),
        case(
            "attention_core",
            "scores[1,8,1408,256] value[1,8,256,1408] out_value[1,8,1408,256]",
            attention_core,
        ),
        case("transpose_head", "x[1,8,1408,256] perm[0,1,3,2]", transpose_head),
        case("concat_heads", "4 x [1,8,1408,64] axis 3", concat_heads),
        case("softmax_op", "x[1,1,128,1408] axis 3", softmax_op),
        case(
            "gemm_plus_gelu",
            "x[1,2048,1,1408] w[6144,2048,1,1] then tanh gelu",
            gemm_plus_gelu,
        ),
        case(
            "conv_dynamic_weight",
            "x[1,256,1,1408] w[128,256,1,1] (input)",
            conv_dynamic_weight,
        ),
        Case {
            requires: &["softmax_op", "conv_dynamic_weight"],
            ..case(
                "attention_core_conv",
                "8 heads x seq 1408 x head_dim 128: q[1,1024,1,1408] k[11264,128,1,1] v[1024,1408,1,1]",
                attention_core_conv,
            )
        },
        case(
            "reshape_matmul",
            "x[1,8,1408,256] -> [8,1408,256], matmul with value[8,256,1408]",
            reshape_matmul,
        ),
    ]);
    cases
}

fn measure(program: &Program, cache: &ProgramCache) -> Result<Measurement> {
    let (mut on_ane, mut off_ane, mut no_device) = (Vec::new(), Vec::new(), 0);
    let model = cache.load_with_placement(
        &program.package,
        ComputeUnits::CpuAndNeuralEngine,
        Duration::from_secs(180),
        |facts| {
            for op in facts.functions.iter().filter(|f| f.name == "main").flat_map(|f| &f.operations) {
                match op.usage.as_ref().and_then(|u| u.preferred.as_ref()) {
                    Some(d) if d.kind == ComputeDeviceKind::NeuralEngine => on_ane.push(op.operator.clone()),
                    Some(d) => off_ane.push(format!("{}:{}", op.operator, d.class_name)),
                    None => no_device += 1,
                }
            }
            Ok(())
        },
    )?;
    let data: Vec<Vec<f16>> = program
        .inputs
        .iter()
        .enumerate()
        .map(|(i, v)| values(v.shape().iter().product(), i * 7919))
        .collect();
    let inputs: Vec<(&str, &[f16], &[usize])> = program
        .inputs
        .iter()
        .zip(&data)
        .map(|(v, d)| (v.name(), &d[..], v.shape()))
        .collect();
    let outputs: Vec<(&str, &[usize])> = program.outputs.iter().map(|v| (v.name(), v.shape())).collect();

    let got = model.predict(&inputs, &outputs)?;
    let finite = got.iter().flatten().all(|v| v.is_finite());
    let nonzero = got.iter().flatten().any(|v| v.to_f32() != 0.0);
    for _ in 0..WARMUP {
        model.predict(&inputs, &outputs)?;
    }
    let mut ts: Vec<f64> = (0..TIMED)
        .map(|_| {
            let t0 = Instant::now();
            model.predict(&inputs, &outputs).map(|_| t0.elapsed().as_secs_f64() * 1e3)
        })
        .collect::<Result<_>>()?;
    ts.sort_by(f64::total_cmp);
    Ok(Measurement {
        on_ane,
        off_ane,
        no_device,
        ms: ts[TIMED / 2],
        ms_min: ts[0],
        ms_max: ts[TIMED - 1],
        finite,
        nonzero,
        note: program.note,
    })
}

fn esc(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            '\n' => vec!['\\', 'n'],
            c if (c as u32) < 0x20 => vec![' '],
            c => vec![c],
        })
        .collect()
}

fn strings(v: &[String]) -> String {
    let items: Vec<String> = v.iter().map(|s| format!("\"{}\"", esc(s))).collect();
    format!("[{}]", items.join(","))
}

fn sysctl(name: &str) -> String {
    std::process::Command::new("sysctl")
        .args(["-n", name])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|| "unknown".into())
}

fn main() -> Result<()> {
    let root = std::env::temp_dir().join(format!("aneops-{}", std::process::id()));
    let (mut measured, mut refused, mut unsupported, mut blocked) = (0, 0, 0, 0);
    let mut all_on_ane: Vec<String> = Vec::new();
    let mut placed: Vec<&str> = Vec::new();
    for (i, case) in cases().iter().enumerate() {
        let head = format!("\"case\":\"{}\",\"shape\":\"{}\"", case.name, esc(&case.shape));
        let t0 = Instant::now();
        let missing: Vec<String> = case
            .requires
            .iter()
            .filter(|r| !placed.contains(r))
            .map(|r| r.to_string())
            .collect();
        let outcome = if !missing.is_empty() {
            Outcome::Blocked(missing)
        } else {
            match (case.author)() {
                Err(e) => Outcome::Unsupported(e.to_string()),
                Ok(program) => {
                    let dir = root.join(i.to_string());
                    let _ = std::fs::remove_dir_all(&dir);
                    let result = measure(&program, &ProgramCache::new(&dir));
                    let _ = std::fs::remove_dir_all(&dir);
                    match result {
                        Ok(m) => Outcome::Measured(m),
                        Err(e) => Outcome::Refused(e.to_string()),
                    }
                }
            }
        };
        let build_s = t0.elapsed().as_secs_f64();
        let line = match outcome {
            Outcome::Blocked(by) => {
                blocked += 1;
                format!("{{{head},\"status\":\"blocked\",\"blocked_by\":{}}}", strings(&by))
            }
            Outcome::Unsupported(reason) => {
                unsupported += 1;
                format!(
                    "{{{head},\"status\":\"unsupported\",\"reason\":\"{}\",\"build_s\":{build_s:.2}}}",
                    esc(&reason)
                )
            }
            Outcome::Refused(reason) => {
                refused += 1;
                format!(
                    "{{{head},\"status\":\"refused\",\"refused\":\"{}\",\"build_s\":{build_s:.2}}}",
                    esc(&reason)
                )
            }
            Outcome::Measured(m) => {
                measured += 1;
                if m.off_ane.is_empty() && !m.on_ane.is_empty() {
                    all_on_ane.push(format!("{} {}", case.name, case.shape));
                    placed.push(case.name);
                }
                format!(
                    "{{{head},\"status\":\"measured\",\"on_ane\":{},\"off_ane\":{},\"on_ane_ops\":{},\"no_device\":{},\"ms\":{:.4},\"ms_min\":{:.4},\"ms_max\":{:.4},\"build_s\":{build_s:.2},\"finite\":{},\"nonzero\":{},\"note\":\"{}\"}}",
                    m.on_ane.len(),
                    strings(&m.off_ane),
                    strings(&m.on_ane),
                    m.no_device,
                    m.ms,
                    m.ms_min,
                    m.ms_max,
                    m.finite,
                    m.nonzero,
                    esc(m.note)
                )
            }
        };
        println!("{line}");
    }
    let _ = std::fs::remove_dir_all(&root);
    println!(
        "{{\"summary\":\"aneops\",\"chip\":\"{}\",\"os_build\":\"{}\",\"compute_units\":\"CpuAndNeuralEngine\",\"warmup\":{WARMUP},\"timed\":{TIMED},\"cases\":{},\"measured\":{measured},\"refused\":{refused},\"unsupported\":{unsupported},\"blocked\":{blocked},\"all_on_ane\":{}}}",
        esc(&sysctl("machdep.cpu.brand_string")),
        esc(&sysctl("kern.osversion")),
        measured + refused + unsupported + blocked,
        strings(&all_on_ane)
    );
    if measured == 0 {
        return Err(Error::Other("aneops measured no case".into()));
    }
    Ok(())
}

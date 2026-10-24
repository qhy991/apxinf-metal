#![cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]

use apxinf_core::Result;
use apxinf_mlx::fusions::Bf16TableRope;
use apxinf_mlx::{Array, Compiled, MlxDType as D, Stream};
use half::bf16;

#[derive(Clone, Copy, Debug)]
enum Values {
    Normal,
    Large,
    SignedZero,
}

#[derive(Clone, Copy, Debug)]
enum Layout {
    Dense,
    TokenMajor,
    Stepped,
    Broadcast,
}

fn upload(s: &Stream, shape: &[usize], values: &[bf16]) -> Result<Array> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| v.to_bits().to_ne_bytes())
        .collect();
    Array::from_bytes(s, shape, D::BF16, &bytes)
}

fn bits(x: &Array) -> Result<Vec<u16>> {
    assert_eq!(x.dtype(), D::BF16);
    Ok(x.to_bytes()?
        .chunks_exact(2)
        .map(|b| u16::from_ne_bytes([b[0], b[1]]))
        .collect())
}

fn data(n: usize, seed: usize, values: Values) -> Vec<bf16> {
    (0..n)
        .map(|i| match values {
            Values::SignedZero => bf16::from_bits(if (i + seed) % 2 == 0 { 0x8000 } else { 0 }),
            Values::Normal | Values::Large => {
                let scale = if matches!(values, Values::Large) {
                    1_048_576.
                } else {
                    1.
                };
                bf16::from_f32((((i * 37 + seed) % 251) as f32 - 125.) / 19. * scale)
            }
        })
        .collect()
}

fn input(
    s: &Stream,
    h: usize,
    l: usize,
    d: usize,
    seed: usize,
    values: Values,
    layout: Layout,
) -> Result<Array> {
    let shape = [1, h, l, d];
    match layout {
        Layout::Dense => upload(s, &shape, &data(h * l * d, seed, values)),
        Layout::TokenMajor => {
            upload(s, &[1, l, h, d], &data(h * l * d, seed, values))?.transpose(&[0, 2, 1, 3])
        }
        Layout::Stepped => {
            let n = h * l * d * 3 + 5;
            upload(s, &[n], &data(n, seed, values))?.as_strided(
                &shape,
                &[0, l * d * 3, d * 3, 3],
                5,
            )
        }
        Layout::Broadcast => upload(s, &[1, 1, 1, d], &data(d, seed, values))?.broadcast_to(&shape),
    }
}

fn table(s: &Stream, l: usize, d: usize, seed: usize, layout: Layout) -> Result<Array> {
    let shape = [1, 1, l, d];
    match layout {
        Layout::Dense => upload(
            s,
            &[1, 1, l + 4, d],
            &data((l + 4) * d, seed, Values::Normal),
        )?
        .slice_axis(2, 2, l + 2),
        Layout::TokenMajor => upload(
            s,
            &[1, 1, d, l + 3],
            &data(d * (l + 3), seed, Values::Normal),
        )?
        .transpose(&[0, 1, 3, 2])?
        .slice_axis(2, 1, l + 1),
        Layout::Stepped => {
            let n = l * d * 3 + 7;
            upload(s, &[n], &data(n, seed, Values::Normal))?.as_strided(
                &shape,
                &[0, 0, d * 3, 2],
                7,
            )
        }
        Layout::Broadcast => {
            upload(s, &[1, 1, 1, d], &data(d, seed, Values::Normal))?.broadcast_to(&shape)
        }
    }
}

fn reference(x: &Array, cos: &Array, sin: &Array) -> Result<Vec<u16>> {
    let (l, d) = (x.shape()[2], x.shape()[3]);
    let x = bits(x)?;
    let cos = bits(cos)?;
    let sin = bits(sin)?;
    Ok(x.iter()
        .enumerate()
        .map(|(i, &value)| {
            let c = i % d;
            let t = i % (l * d);
            let partner = if c < d / 2 { i + d / 2 } else { i - d / 2 };
            let rotated = x[partner] ^ if c < d / 2 { 0x8000 } else { 0 };
            let left =
                bf16::from_f32(bf16::from_bits(value).to_f32() * bf16::from_bits(cos[t]).to_f32());
            let right = bf16::from_f32(
                bf16::from_bits(rotated).to_f32() * bf16::from_bits(sin[t]).to_f32(),
            );
            bf16::from_f32(left.to_f32() + right.to_f32()).to_bits()
        })
        .collect())
}

fn ordered(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let d = x.shape()[3];
    let half = Array::concat(
        &[
            &x.slice_axis(3, d / 2, d)?.neg()?,
            &x.slice_axis(3, 0, d / 2)?,
        ],
        3,
    )?;
    x.mul(cos)?.add(&half.mul(sin)?)
}

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn multirow_rope_matches_host_and_ordered_bits_across_views() -> Result<()> {
    let s = Stream::metal(0)?;
    let kernel = Bf16TableRope::new(&s)?;
    let composed = Compiled::new(&s, 1, |a| Ok(vec![ordered(&a[0], &a[1], &a[2])?]))?;
    for (h, l, d) in [
        (2, 1, 8),
        (1, 2, 8),
        (2, 3, 8),
        (8, 3, 8),
        (16, 2, 8),
        (1, 3, 128),
        (2, 2, 128),
        (8, 3, 128),
        (16, 2, 128),
        (8, 1023, 128),
        (16, 2048, 128),
        (8, 2047, 128),
    ] {
        for values in [Values::Normal, Values::Large, Values::SignedZero] {
            for layout in [
                Layout::Dense,
                Layout::TokenMajor,
                Layout::Stepped,
                Layout::Broadcast,
            ] {
                let x = input(&s, h, l, d, 11, values, layout)?;
                let cos = table(&s, l, d, 23, Layout::TokenMajor)?;
                let sin = table(&s, l, d, 47, Layout::Stepped)?;
                let args = [x, cos, sin];
                let before = args.iter().map(bits).collect::<Result<Vec<_>>>()?;
                if matches!(values, Values::SignedZero) {
                    assert!(before[0].contains(&0) && before[0].contains(&0x8000));
                }
                assert!(before[1].chunks(d).any(|r| r[..d / 2] != r[d / 2..]));
                assert!(before[2].chunks(d).any(|r| r[..d / 2] != r[d / 2..]));
                let expected = reference(&args[0], &args[1], &args[2])?;
                let output = kernel.call(&args[0], &args[1], &args[2])?;
                assert_eq!((output.shape(), output.dtype()), (args[0].shape(), D::BF16));
                assert_eq!(
                    bits(&output)?,
                    expected,
                    "H{h} L{l} D{d} {values:?} {layout:?}"
                );
                assert_eq!(
                    bits(&composed.call_and_eval(&args)?.remove(0))?,
                    expected,
                    "ordered H{h} L{l} D{d}"
                );
                let linear = output.as_strided(&[h * l * d], &[1], 0)?;
                assert_eq!(bits(&linear)?, expected);
                assert_eq!(args.iter().map(bits).collect::<Result<Vec<_>>>()?, before);
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn multirow_rope_fixed_fixture_requires_separate_bf16_products() -> Result<()> {
    let s = Stream::metal(0)?;
    let a = bf16::from_bits(0x3f02).to_f32();
    let b = bf16::from_bits(0x3f01).to_f32();
    let delayed = bf16::from_f32(a * a + (-a) * b).to_bits();
    assert_eq!(delayed, 0x3b02);
    for dim in [8, 128] {
        let x = upload(&s, &[1, 2, 3, dim], &vec![bf16::from_bits(0x3f02); 6 * dim])?;
        let cos = upload(&s, &[1, 1, 3, dim], &vec![bf16::from_bits(0x3f02); 3 * dim])?;
        let sin = upload(&s, &[1, 1, 3, dim], &vec![bf16::from_bits(0x3f01); 3 * dim])?;
        let expected = reference(&x, &cos, &sin)?;
        for row in expected.chunks(dim) {
            assert!(row[..dim / 2].iter().all(|&value| value == 0x3b00));
        }
        let output = Bf16TableRope::new(&s)?.call(&x, &cos, &sin)?;
        assert_eq!(bits(&output)?, expected, "fixed fixture D{dim}");
        assert_eq!(bits(&ordered(&x, &cos, &sin)?)?, expected);
    }
    Ok(())
}

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn multirow_rope_compiled_replay_changes_operands_strides_and_lifetimes() -> Result<()> {
    let s = Stream::metal(0)?;
    for (h, l, d) in [(2, 3, 8), (8, 1023, 128), (16, 2047, 128)] {
        let kernel = Bf16TableRope::new(&s)?;
        let compiled = Compiled::new(&s, 1, move |a| Ok(vec![kernel.call(&a[0], &a[1], &a[2])?]))?;
        let mut args = [
            input(&s, h, l, d, 11, Values::Normal, Layout::TokenMajor)?,
            table(&s, l, d, 23, Layout::Dense)?,
            table(&s, l, d, 47, Layout::Stepped)?,
        ];
        let first = compiled.prepare(&args)?;
        let first_bits = bits(&first[0])?;
        let traces = s.counters().trace_callbacks;
        let replacements = [
            input(&s, h, l, d, 61, Values::Normal, Layout::Stepped)?,
            table(&s, l, d, 79, Layout::TokenMajor)?,
            table(&s, l, d, 97, Layout::Broadcast)?,
        ];
        let mut previous = first_bits.clone();
        for (operand, replacement) in replacements.into_iter().enumerate() {
            assert_eq!(replacement.shape(), args[operand].shape());
            args[operand] = replacement;
            let actual = bits(&compiled.call_and_eval(&args)?.remove(0))?;
            assert_eq!(s.counters().trace_callbacks, traces);
            assert_eq!(
                actual,
                reference(&args[0], &args[1], &args[2])?,
                "changed operand {operand}"
            );
            assert_ne!(actual, previous, "operand {operand} must affect output");
            assert_eq!(bits(&first[0])?, first_bits);
            previous = actual;
        }
        args = [
            input(&s, h, l, d, 107, Values::Normal, Layout::Broadcast)?,
            table(&s, l, d, 131, Layout::Stepped)?,
            table(&s, l, d, 149, Layout::Dense)?,
        ];
        let expected = reference(&args[0], &args[1], &args[2])?;
        let pending = compiled.call(&args)?;
        assert_eq!(s.counters().trace_callbacks, traces);
        drop(compiled);
        drop(args);
        assert_eq!(bits(&pending[0])?, expected);
        assert_eq!(bits(&first[0])?, first_bits);
    }
    let pending = {
        let kernel = Bf16TableRope::new(&s)?;
        let x = input(&s, 2, 3, 8, 11, Values::Normal, Layout::Stepped)?;
        let cos = table(&s, 3, 8, 23, Layout::TokenMajor)?;
        let sin = table(&s, 3, 8, 47, Layout::Stepped)?;
        (kernel.call(&x, &cos, &sin)?, reference(&x, &cos, &sin)?)
    };
    assert_eq!(bits(&pending.0)?, pending.1);
    Ok(())
}

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn multirow_rope_rejects_geometry_dtype_extent_and_foreign_streams() -> Result<()> {
    let s = Stream::metal(0)?;
    let kernel = Bf16TableRope::new(&s)?;
    let args = [
        Array::zeros(&s, &[1, 2, 3, 8], D::BF16)?,
        Array::zeros(&s, &[1, 1, 3, 8], D::BF16)?,
        Array::zeros(&s, &[1, 1, 3, 8], D::BF16)?,
    ];
    for operand in 0..3 {
        let mut invalid = args.clone();
        invalid[operand] = invalid[operand].cast(D::F32)?;
        assert!(kernel.call(&invalid[0], &invalid[1], &invalid[2]).is_err());
        let foreign = Stream::metal(0)?;
        invalid[operand] = Array::zeros(&foreign, args[operand].shape(), D::BF16)?;
        assert!(kernel.call(&invalid[0], &invalid[1], &invalid[2]).is_err());
    }
    for (operand, shape) in [
        (0, vec![2, 3, 8]),
        (0, vec![2, 2, 3, 8]),
        (0, vec![1, 0, 3, 8]),
        (0, vec![1, 2, 0, 8]),
        (0, vec![1, 2, 3, 0]),
        (0, vec![1, 2, 3, 7]),
        (1, vec![8]),
        (1, vec![1, 2, 3, 8]),
        (1, vec![1, 1, 2, 8]),
        (2, vec![1, 1, 3, 6]),
        (2, vec![1, 1, 4, 8]),
        (2, vec![2, 1, 3, 8]),
    ] {
        let mut invalid = args.clone();
        invalid[operand] = Array::zeros(&s, &shape, D::BF16)?;
        assert!(
            kernel.call(&invalid[0], &invalid[1], &invalid[2]).is_err(),
            "operand {operand}, {shape:?}"
        );
    }
    let scalar = Array::scalar(&s, 0., D::BF16)?;
    let huge = scalar.broadcast_to(&[1, i32::MAX as usize, 2, 2])?;
    let small = scalar.broadcast_to(&[1, 1, 2, 2])?;
    assert!(kernel.call(&huge, &small, &small).is_err());
    Ok(())
}

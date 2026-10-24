use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

/// Half-split BF16 rotation with explicit, independently strided cosine and sine.
/// Negation, each product and each sum round separately to BF16.
pub struct Bf16TableRope {
    kernel: MetalKernel,
}

fn checked_elements(heads: usize, rows: usize, dim: usize) -> Result<usize> {
    if heads == 0 || rows == 0 || dim == 0 || dim % 2 != 0 {
        return Err(Error::Contract(
            "table RoPE requires positive H/L and positive even D",
        ));
    }
    heads
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(dim))
        .filter(|&n| i32::try_from(n).is_ok())
        .ok_or(Error::Contract("table RoPE element count exceeds i32"))
}

impl Bf16TableRope {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new_strided(
                stream,
                "apxinf_bf16_table_rope",
                &["x", "cos", "sin"],
                &["out"],
                include_str!("../native/bf16_table_rope.metal"),
                "",
            )?,
        })
    }

    /// Accepts BF16 X `[1,H,L,D]` and BF16 tables `[1,1,L,D]`.
    /// H/L must be positive, D positive and even, and H*L*D at most i32::MAX.
    /// Returns a fresh contiguous array with X's shape. Table halves may differ.
    pub fn call(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let [1, heads, rows, dim] = *x.shape() else {
            return Err(Error::Contract("table RoPE X must be [1,H,L,D]"));
        };
        let elements = checked_elements(heads, rows, dim)?;
        if [x, cos, sin].iter().any(|a| a.dtype() != MlxDType::BF16)
            || [cos, sin].iter().any(|a| a.shape() != [1, 1, rows, dim])
        {
            return Err(Error::Contract(
                "table RoPE requires BF16 X and [1,1,L,D] tables",
            ));
        }
        // SAFETY: each guarded pair thread owns columns c and c+D/2 in one
        // (head,row), so every output column is written once. Checked E bounds
        // output offsets, launch and template products by i32::MAX. Valid Array
        // views establish input storage bounds; signed int64 coordinate/stride
        // products preserve their layouts and already-applied slice offsets.
        // MetalKernel checks that every input belongs to this stream.
        let mut output = unsafe {
            self.kernel.call(
                &[x.clone(), cos.clone(), sin.clone()],
                &[(x.shape(), MlxDType::BF16)],
                [elements / 2, 1, 1],
                [256, 1, 1],
                &[
                    ("T", TemplateArg::DType(MlxDType::BF16)),
                    ("HEADS", TemplateArg::Int(heads as i32)),
                    ("ROWS", TemplateArg::Int(rows as i32)),
                    ("DIM", TemplateArg::Int(dim as i32)),
                ],
            )?
        };
        Ok(output.pop().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::checked_elements;

    #[test]
    fn geometry_rejects_zero_odd_and_unsigned_overflow_without_arrays() {
        assert_eq!(checked_elements(16, 2048, 128).unwrap(), 4_194_304);
        assert_eq!(
            checked_elements(1, 1, i32::MAX as usize - 1).unwrap(),
            i32::MAX as usize - 1
        );
        for (h, l, d) in [
            (0, 2, 8),
            (2, 0, 8),
            (2, 2, 0),
            (2, 2, 7),
            (1, 1, i32::MAX as usize + 1),
            (i32::MAX as usize, 2, 2),
            (usize::MAX, 2, 2),
            (2, usize::MAX, 2),
            (2, 1, usize::MAX - 1),
        ] {
            assert!(checked_elements(h, l, d).is_err(), "H{h} L{l} D{d}");
        }
    }
}

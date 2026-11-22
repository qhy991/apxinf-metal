//! WallOSS dynamic FP8 E4M3 encoding helpers.

/// Largest finite NVIDIA/CUDA E4M3 value (`0x7e`).
pub const E4M3_MAX: f32 = 448.0;

/// CUDA-compatible saturating finite E4M3 encoding, round-to-nearest-even.
pub fn encode_e4m3(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let value = value.abs();
    if value == 0.0 {
        return sign;
    }
    if !value.is_finite() || value >= E4M3_MAX {
        return sign | 0x7e;
    }

    // Subnormals have a fixed 2^-9 quantum.
    if value < 2f32.powi(-6) {
        let mantissa = round_ties_even(value * 512.0).min(7) as u8;
        return sign | mantissa;
    }

    let exponent = value.log2().floor() as i32;
    let mut exponent_bits = exponent + 7;
    let normalized = value / 2f32.powi(exponent) - 1.0;
    let mut mantissa = round_ties_even(normalized * 8.0) as i32;
    if mantissa == 8 {
        exponent_bits += 1;
        mantissa = 0;
    }
    if exponent_bits > 15 || (exponent_bits == 15 && mantissa >= 7) {
        return sign | 0x7e;
    }
    sign | ((exponent_bits as u8) << 3) | mantissa as u8
}

#[cfg(test)]
pub fn decode_e4m3(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    if exponent == 0 {
        return sign * mantissa as f32 * 2f32.powi(-9);
    }
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    sign * (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exponent as i32 - 7)
}

fn round_ties_even(value: f32) -> u32 {
    let floor = value.floor();
    let fraction = value - floor;
    let floor_u = floor as u32;
    if fraction > 0.5 || (fraction == 0.5 && floor_u & 1 == 1) {
        floor_u + 1
    } else {
        floor_u
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_e4m3_values_match_cuda_layout() {
        assert_eq!(encode_e4m3(0.0), 0x00);
        assert_eq!(encode_e4m3(-0.0), 0x80);
        assert_eq!(encode_e4m3(1.0), 0x38);
        assert_eq!(encode_e4m3(-1.0), 0xb8);
        assert_eq!(encode_e4m3(448.0), 0x7e);
        assert_eq!(decode_e4m3(0x01), 2f32.powi(-9));
        assert_eq!(decode_e4m3(0x7e), 448.0);
        assert!(decode_e4m3(0x7f).is_nan());
    }

    #[test]
    fn encoding_saturates_and_rounds_ties_to_even() {
        assert_eq!(encode_e4m3(f32::INFINITY), 0x7e);
        assert_eq!(encode_e4m3(1000.0), 0x7e);
        // Halfway between 1.0 (mantissa 0, even) and 1.125 (mantissa 1).
        assert_eq!(encode_e4m3(1.0625), 0x38);
        // Halfway between mantissas 1 and 2 rounds to mantissa 2.
        assert_eq!(encode_e4m3(1.1875), 0x3a);
    }
}

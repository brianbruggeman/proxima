use super::*;
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

const QK: usize = 256;

struct ReferenceQ8k {
    d: f32,
    qs: [i8; QK],
    bsums: [i16; 16],
}

fn reference_nearest_int(value: f32) -> i32 {
    let bits = (value + 12_582_912.0).to_bits();
    (bits & 0x007f_ffff) as i32 - 0x0040_0000
}

fn reference_quantize_q8_k(x: &[f32]) -> ReferenceQ8k {
    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for &value in x {
        if value.abs() > amax {
            amax = value.abs();
            max = value;
        }
    }
    let mut out = ReferenceQ8k {
        d: 0.0,
        qs: [0; QK],
        bsums: [0; 16],
    };
    if amax == 0.0 {
        return out;
    }
    let iscale = -127.0f32 / max;
    for (slot, &value) in out.qs.iter_mut().zip(x) {
        *slot = reference_nearest_int(iscale * value).min(127) as i8;
    }
    for (group, sum) in out.bsums.iter_mut().enumerate() {
        *sum = out.qs[group * 16..group * 16 + 16]
            .iter()
            .map(|&level| i16::from(level))
            .sum();
    }
    out.d = 1.0 / iscale;
    out
}

fn half_at(bytes: &[u8], offset: usize) -> f32 {
    half::f16::from_le_bytes([bytes[offset], bytes[offset + 1]]).to_f32()
}

fn reference_dot_q4_k(block: &[u8], act: &ReferenceQ8k) -> (f64, f64) {
    let mut utmp = [0u32; 4];
    for (slot, bytes) in utmp.iter_mut().zip(block[4..16].as_chunks::<4>().0) {
        *slot = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }
    utmp[3] = ((utmp[2] >> 4) & 0x0f0f_0f0f) | (((utmp[1] >> 6) & 0x0303_0303) << 4);
    let uaux = utmp[1] & 0x3f3f_3f3f;
    utmp[1] = (utmp[2] & 0x0f0f_0f0f) | (((utmp[0] >> 6) & 0x0303_0303) << 4);
    utmp[2] = uaux;
    utmp[0] &= 0x3f3f_3f3f;
    let mut codes = [0u8; 16];
    for (index, word) in utmp.iter().enumerate() {
        codes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    let (scales, mins) = codes.split_at(8);

    let mut aux8 = [0i32; QK];
    let q4 = &block[16..144];
    for j in 0..QK / 64 {
        for l in 0..32 {
            aux8[j * 64 + l] = i32::from(q4[j * 32 + l] & 0x0f);
            aux8[j * 64 + 32 + l] = i32::from(q4[j * 32 + l] >> 4);
        }
    }
    let mut sumi_mins = 0i32;
    for j in 0..16 {
        sumi_mins += i32::from(act.bsums[j]) * i32::from(mins[j / 2]);
    }
    let mut aux32 = [0i32; 8];
    for j in 0..QK / 32 {
        let scale = i32::from(scales[j]);
        for l in 0..32 {
            aux32[l % 8] += scale * i32::from(act.qs[j * 32 + l]) * aux8[j * 32 + l];
        }
    }
    let d = half_at(block, 0) * act.d;
    let dmin = half_at(block, 2) * act.d;
    let lane_abs: f64 = aux32.iter().map(|&lane| f64::from(lane).abs()).sum();
    let lane_sum: f64 = aux32.iter().map(|&lane| f64::from(lane)).sum();
    let value = lane_sum * f64::from(d) - f64::from(dmin) * f64::from(sumi_mins);
    let magnitude =
        lane_abs * f64::from(d).abs() + f64::from(dmin).abs() * f64::from(sumi_mins).abs();
    (value, magnitude)
}

fn reference_dot_q6_k(block: &[u8], act: &ReferenceQ8k) -> (f64, f64) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let mut aux8 = [0i32; QK];
    for half in 0..2 {
        for l in 0..32 {
            let base = half * 128;
            let q4 = &ql[half * 64..];
            let high = qh[half * 32 + l];
            aux8[base + l] = i32::from((q4[l] & 0x0f) | ((high & 3) << 4)) - 32;
            aux8[base + l + 32] = i32::from((q4[l + 32] & 0x0f) | (((high >> 2) & 3) << 4)) - 32;
            aux8[base + l + 64] = i32::from((q4[l] >> 4) | (((high >> 4) & 3) << 4)) - 32;
            aux8[base + l + 96] = i32::from((q4[l + 32] >> 4) | (((high >> 6) & 3) << 4)) - 32;
        }
    }
    let mut aux32 = [0i32; 8];
    for j in 0..16 {
        let scale = i32::from(scales[j] as i8);
        for l in 0..16 {
            aux32[l % 8] += scale * i32::from(act.qs[j * 16 + l]) * aux8[j * 16 + l];
        }
    }
    let d = half_at(block, 208) * act.d;
    let magnitude: f64 =
        aux32.iter().map(|&lane| f64::from(lane).abs()).sum::<f64>() * f64::from(d).abs();
    let value: f64 = aux32.iter().map(|&lane| f64::from(lane)).sum::<f64>() * f64::from(d);
    (value, magnitude)
}

fn finite_scale_block(mut bytes: Vec<u8>, scale_offsets: &[usize]) -> Vec<u8> {
    for &offset in scale_offsets {
        let scale = half::f16::from_f32(f32::from(bytes[offset]) / 64.0 + 0.01);
        let encoded = scale.to_le_bytes();
        bytes[offset] = encoded[0];
        bytes[offset + 1] = encoded[1];
    }
    bytes
}

fn activation_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        prop_oneof![8 => -2.0f32..2.0f32, 1 => -300.0f32..300.0f32, 1 => Just(0.0f32)],
        QK,
    )
}

fn relative_close(left: f32, right: f64, magnitude: f64) -> bool {
    (f64::from(left) - right).abs() <= 2e-6 * magnitude.max(1e-6)
}

proptest! {
    #[test]
    fn activation_quantize_matches_llama_reference(x in activation_strategy()) {
        let mut ours = vec![0u8; Q8K_BLOCK_BYTES];
        prop_assert!(quantize_row_q8k(&x, &mut ours).is_ok());
        let reference = reference_quantize_q8_k(&x);
        let ours_d = f32::from_le_bytes([ours[0], ours[1], ours[2], ours[3]]);
        prop_assert_eq!(ours_d.to_bits(), reference.d.to_bits());
        for index in 0..QK {
            prop_assert_eq!(ours[4 + index] as i8, reference.qs[index], "qs[{}]", index);
        }
        for group in 0..16 {
            let offset = 4 + QK + group * 2;
            let bsum = i16::from_le_bytes([ours[offset], ours[offset + 1]]);
            prop_assert_eq!(bsum, reference.bsums[group], "bsums[{}]", group);
        }
    }

    #[test]
    fn q4_k_int8_arms_match_llama_reference(
        weights in prop::collection::vec(any::<u8>(), 144),
        x in activation_strategy(),
    ) {
        let block = finite_scale_block(weights, &[0, 2]);
        let mut packed = vec![0u8; Q8K_BLOCK_BYTES];
        prop_assert!(quantize_row_q8k(&x, &mut packed).is_ok());
        let reference_act = reference_quantize_q8_k(&x);
        let (expected, magnitude) = reference_dot_q4_k(&block, &reference_act);
        let runtime = dot_q4k_q8k(&block, &packed).map_err(|error| TestCaseError::fail(format!("{error:?}")))?;
        let portable = dot_q4k_q8k_portable(&block, &packed).map_err(|error| TestCaseError::fail(format!("{error:?}")))?;
        prop_assert!(relative_close(runtime, expected, magnitude), "runtime {} vs reference {}", runtime, expected);
        prop_assert!(relative_close(portable, expected, magnitude), "portable {} vs reference {}", portable, expected);
    }

    #[test]
    fn q6_k_int8_arms_match_llama_reference(
        weights in prop::collection::vec(any::<u8>(), 210),
        x in activation_strategy(),
    ) {
        let block = finite_scale_block(weights, &[208]);
        let mut packed = vec![0u8; Q8K_BLOCK_BYTES];
        prop_assert!(quantize_row_q8k(&x, &mut packed).is_ok());
        let reference_act = reference_quantize_q8_k(&x);
        let (expected, magnitude) = reference_dot_q6_k(&block, &reference_act);
        let runtime = dot_q6k_q8k(&block, &packed).map_err(|error| TestCaseError::fail(format!("{error:?}")))?;
        let portable = dot_q6k_q8k_portable(&block, &packed).map_err(|error| TestCaseError::fail(format!("{error:?}")))?;
        prop_assert!(relative_close(runtime, expected, magnitude), "runtime {} vs reference {}", runtime, expected);
        prop_assert!(relative_close(portable, expected, magnitude), "portable {} vs reference {}", portable, expected);
    }
}

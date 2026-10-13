use std::collections::BTreeMap;

use proxima::ProximaError;
use proxima_safetensors::sized::HEADER_LEN_BYTES;
use proxima_safetensors::{
    SafetensorsModel, TensorEntry, TensorPayload, dtype_to_wire, parse_complete, write_complete,
};
use proxima_tensor::DType;

pub const PLAN_KEY: &str = "plan";

pub type TensorRecord = (String, Vec<u64>, Vec<f32>);

pub fn encode_request(
    plan_toml: &str,
    inputs: &[(&str, &[u64], &[f32])],
) -> Result<Vec<u8>, ProximaError> {
    let metadata = BTreeMap::from([(PLAN_KEY.to_string(), plan_toml.to_string())]);
    write_f32_model(metadata, inputs)
}

pub fn decode_request(bytes: &[u8]) -> Result<(String, Vec<TensorRecord>), ProximaError> {
    let (mut metadata, tensors) = read_f32_model(bytes)?;
    let plan = metadata
        .remove(PLAN_KEY)
        .ok_or_else(|| ProximaError::Decode(format!("request metadata has no `{PLAN_KEY}` key")))?;
    Ok((plan, tensors))
}

pub fn encode_response(tensors: &[TensorRecord]) -> Result<Vec<u8>, ProximaError> {
    let borrowed: Vec<(&str, &[u64], &[f32])> = tensors
        .iter()
        .map(|(name, shape, values)| (name.as_str(), shape.as_slice(), values.as_slice()))
        .collect();
    write_f32_model(BTreeMap::new(), &borrowed)
}

pub fn decode_response(bytes: &[u8]) -> Result<Vec<TensorRecord>, ProximaError> {
    read_f32_model(bytes).map(|(_, tensors)| tensors)
}

fn write_f32_model(
    metadata: BTreeMap<String, String>,
    tensors: &[(&str, &[u64], &[f32])],
) -> Result<Vec<u8>, ProximaError> {
    let owned: Vec<(&str, &[u64], Vec<u8>)> = tensors
        .iter()
        .map(|(name, shape, values)| (*name, *shape, le_bytes(values)))
        .collect();
    let payloads = owned
        .iter()
        .map(|(name, shape, data)| TensorPayload {
            name: (*name).to_string(),
            dtype: DType::Float32,
            shape: shape.to_vec(),
            data,
        })
        .collect();
    write_complete(&SafetensorsModel {
        tensors: payloads,
        metadata,
    })
    .map_err(|error| ProximaError::Encode(format!("tensor body could not be written: {error}")))
}

fn le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn read_f32_model(
    bytes: &[u8],
) -> Result<(BTreeMap<String, String>, Vec<TensorRecord>), ProximaError> {
    let manifest = parse_complete(bytes)
        .map_err(|error| ProximaError::Decode(format!("body is not safetensors: {error}")))?;
    let data = data_region(bytes)?;
    let tensors = manifest
        .tensors
        .iter()
        .map(|entry| read_f32_entry(entry, data))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((manifest.metadata, tensors))
}

fn data_region(bytes: &[u8]) -> Result<&[u8], ProximaError> {
    let prefix = bytes.get(..HEADER_LEN_BYTES).ok_or_else(|| {
        ProximaError::Decode("body is shorter than its header length".to_string())
    })?;
    let header_len = u64::from_le_bytes(
        <[u8; HEADER_LEN_BYTES]>::try_from(prefix)
            .map_err(|error| ProximaError::Decode(error.to_string()))?,
    );
    let header_end = usize::try_from(header_len)
        .ok()
        .and_then(|len| len.checked_add(HEADER_LEN_BYTES))
        .ok_or_else(|| ProximaError::Decode("header length overflows the body".to_string()))?;
    bytes
        .get(header_end..)
        .ok_or_else(|| ProximaError::Decode("body is shorter than its header".to_string()))
}

fn read_f32_entry(entry: &TensorEntry, data: &[u8]) -> Result<TensorRecord, ProximaError> {
    if entry.dtype != DType::Float32 {
        let wire = dtype_to_wire(entry.dtype).unwrap_or("unsupported");
        return Err(ProximaError::Config(format!(
            "tensor `{}` has dtype {wire}; the tensor wire carries only F32",
            entry.name
        )));
    }
    let (begin, end) = entry.data_offsets;
    let raw = usize::try_from(begin)
        .ok()
        .zip(usize::try_from(end).ok())
        .and_then(|(begin, end)| data.get(begin..end))
        .ok_or_else(|| {
            ProximaError::Decode(format!(
                "tensor `{}` data lies outside the body",
                entry.name
            ))
        })?;
    let elements: u64 = entry.shape.iter().product();
    if elements.saturating_mul(4) != raw.len() as u64 {
        return Err(ProximaError::Decode(format!(
            "tensor `{}` declares shape {:?} but carries {} bytes",
            entry.name,
            entry.shape,
            raw.len()
        )));
    }
    let values = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    Ok((entry.name.clone(), entry.shape.clone(), values))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use proxima_safetensors::{SafetensorsModel, TensorPayload, write_complete};
    use proxima_tensor::DType;
    use proxima_tensor::spec::ProgramSpec;
    use proxima_tensor::test_support::Lcg;

    use super::{TensorRecord, decode_request, decode_response, encode_request, encode_response};
    use proxima::ProximaError;

    const PLAN_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../proxima-tensor/src/spec/fixtures/matmul_silu_softmax.toml"
    );

    fn plan_text() -> String {
        std::fs::read_to_string(PLAN_PATH).expect("matmul-silu-softmax fixture is readable")
    }

    fn find_tensor<'a>(tensors: &'a [TensorRecord], name: &str) -> (&'a [u64], &'a [f32]) {
        tensors
            .iter()
            .find(|(record_name, _, _)| record_name == name)
            .map(|(_, shape, values)| (shape.as_slice(), values.as_slice()))
            .expect("the named tensor is in the response")
    }

    fn matmul_inputs() -> (Vec<f32>, Vec<f32>) {
        let mut lcg = Lcg(0x5eed_0003);
        let x: Vec<f32> = (0..8).map(|_| lcg.next_unit() - 0.5).collect();
        let w: Vec<f32> = (0..12).map(|_| lcg.next_unit() - 0.5).collect();
        (x, w)
    }

    #[test]
    fn request_roundtrips_matmul_silu_softmax_plan_and_inputs() {
        let plan = plan_text();
        let (x, w) = matmul_inputs();
        let x_shape = [2_u64, 4];
        let w_shape = [4_u64, 3];
        let bytes = encode_request(&plan, &[("x", &x_shape, &x), ("w", &w_shape, &w)])
            .expect("request encodes");

        let (decoded_plan, tensors) = decode_request(&bytes).expect("request decodes");

        assert_eq!(
            decoded_plan, plan,
            "the plan text survives metadata unchanged"
        );
        assert_eq!(tensors.len(), 2, "one tensor per input was sent");
        let (x_back_shape, x_back) = find_tensor(&tensors, "x");
        let (w_back_shape, w_back) = find_tensor(&tensors, "w");
        assert_eq!(x_back_shape, x_shape.as_slice(), "x keeps its 2x4 shape");
        assert_eq!(w_back_shape, w_shape.as_slice(), "w keeps its 4x3 shape");
        let sent: Vec<u32> = x.iter().chain(&w).map(|value| value.to_bits()).collect();
        let received: Vec<u32> = x_back
            .iter()
            .chain(w_back)
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(received.len(), 20, "twenty floats were sent");
        assert_eq!(received, sent, "every float round-trips bit-for-bit");
    }

    #[test]
    fn response_roundtrips_six_floats() {
        let plan = plan_text();
        let (x, w) = matmul_inputs();
        let x_shape = [2_u64, 4];
        let w_shape = [4_u64, 3];
        let request = encode_request(&plan, &[("x", &x_shape, &x), ("w", &w_shape, &w)])
            .expect("request encodes");
        let (decoded_plan, tensors) = decode_request(&request).expect("request decodes");
        let spec: ProgramSpec = toml::from_str(&decoded_plan).expect("plan parses");
        let inputs: Vec<(&str, &[f32])> = tensors
            .iter()
            .map(|(name, _, values)| (name.as_str(), values.as_slice()))
            .collect();
        let outputs = spec.evaluate_cpu(&inputs).expect("plan evaluates on cpu");

        let response = encode_response(&outputs).expect("response encodes");
        let decoded = decode_response(&response).expect("response decodes");

        let direct = spec
            .evaluate_cpu(&[("x", &x), ("w", &w)])
            .expect("direct evaluation succeeds");
        assert_eq!(decoded.len(), 1, "the fixture declares one output");
        let (name, shape, values) = &decoded[0];
        assert_eq!(name, "softmax");
        assert_eq!(shape.as_slice(), &[2_u64, 3], "output is the 2x3 product");
        let (_, _, direct_values) = &direct[0];
        let received: Vec<u32> = values.iter().map(|value| value.to_bits()).collect();
        let expected: Vec<u32> = direct_values.iter().map(|value| value.to_bits()).collect();
        assert_eq!(received.len(), 6, "six output floats were sent back");
        assert_eq!(
            received, expected,
            "remote output is bit-equal to in-process evaluate_cpu"
        );
    }

    #[test]
    fn decode_request_rejects_missing_plan_metadata() {
        let (x, _) = matmul_inputs();
        let shape = [2_u64, 4];
        let data: Vec<u8> = x.iter().flat_map(|value| value.to_le_bytes()).collect();
        let bytes = write_complete(&SafetensorsModel {
            tensors: vec![TensorPayload {
                name: "x".to_string(),
                dtype: DType::Float32,
                shape: shape.to_vec(),
                data: &data,
            }],
            metadata: std::collections::BTreeMap::new(),
        })
        .expect("body without a plan writes");

        let error = decode_request(&bytes).expect_err("a body with no plan is rejected");

        assert!(
            matches!(&error, ProximaError::Decode(message) if message.contains("plan")),
            "error names the missing plan key: {error}"
        );
    }

    #[test]
    fn decode_request_rejects_truncated_bytes() {
        let plan = plan_text();
        let (x, w) = matmul_inputs();
        let x_shape = [2_u64, 4];
        let w_shape = [4_u64, 3];
        let bytes = encode_request(&plan, &[("x", &x_shape, &x), ("w", &w_shape, &w)])
            .expect("request encodes");
        let truncated = &bytes[..bytes.len() - 4];

        let error = decode_request(truncated).expect_err("a body cut short is rejected");

        assert!(
            matches!(&error, ProximaError::Decode(_)),
            "truncation is a decode error: {error}"
        );
    }

    #[test]
    fn decode_request_rejects_non_f32_tensor() {
        let data = [0_u8; 8];
        let bytes = write_complete(&SafetensorsModel {
            tensors: vec![TensorPayload {
                name: "x".to_string(),
                dtype: DType::Int32,
                shape: vec![2],
                data: &data,
            }],
            metadata: std::collections::BTreeMap::from([(
                super::PLAN_KEY.to_string(),
                plan_text(),
            )]),
        })
        .expect("int32 body writes");

        let error = decode_request(&bytes).expect_err("an int32 tensor is rejected");

        assert!(
            matches!(&error, ProximaError::Config(message) if message.contains("I32")),
            "error names the rejected dtype: {error}"
        );
    }
}

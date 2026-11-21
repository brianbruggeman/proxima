use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};

use proxima::{
    Bytes, PipeHandle, ProximaError, Request, Response, RoutingPipe, SendPipe, into_handle,
};
use proxima_cli::tensor_wire::{TensorRecord, decode_request, encode_response};
use omega::backend::{execute_plan_named, plan_named_with_config};
use omega::config::RuntimeConfig;
use proxima_telemetry::info;
use proxima_tensor::QuantizedBlock;
use proxima_tensor::spec::ProgramSpec;

const TENSOR_RUN_PATH: &str = "/tensor/run";
const SAFETENSORS_CONTENT_TYPE: &str = "application/x-safetensors";

#[derive(Default)]
pub struct TensorRunPipe {
    served: AtomicU64,
}

impl SendPipe for TensorRunPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let (_meta, body) = request.body_bytes().await?;
            let (input_count, outputs) = match evaluate_body(&body) {
                Ok(evaluated) => evaluated,
                Err(message) => return Ok(Response::new(400).with_body(message)),
            };
            let payload = encode_response(&outputs)?;
            let requests_served = self.served.fetch_add(1, Ordering::Relaxed) + 1;
            info!(
                event = "tensor_run",
                requests_served = requests_served,
                inputs = input_count,
                outputs = outputs.len(),
                "tensor plan served"
            );
            Ok(Response::ok(payload).with_header("content-type", SAFETENSORS_CONTENT_TYPE))
        }
    }
}

fn evaluate_body(body: &[u8]) -> Result<(usize, Vec<TensorRecord>), String> {
    let (plan, tensors) = decode_request(body).map_err(|error| error.to_string())?;
    let spec: ProgramSpec =
        toml::from_str(&plan).map_err(|error| format!("plan is not a tensor program: {error}"))?;
    let inputs: Vec<(&str, &[f32])> = tensors
        .iter()
        .map(|(name, _, values)| (name.as_str(), values.as_slice()))
        .collect();
    let config = RuntimeConfig::from_env()
        .map_err(|error| format!("omega runtime configuration is invalid: {error}"))?;
    let outputs = evaluate(&spec, &inputs, &config)?;
    Ok((inputs.len(), outputs))
}

/// Runs the plan on the cpu evaluator unless `config` selects the gpu engine, in
/// which case omega plans and executes it on the configured driver.
fn evaluate(spec: &ProgramSpec, inputs: &[(&str, &[f32])], config: &RuntimeConfig) -> Result<Vec<TensorRecord>, String> {
    if config.engine == "gpu" {
        return evaluate_through_omega(spec, inputs, config);
    }
    spec.evaluate_cpu(inputs)
        .map_err(|error| format!("plan did not evaluate: {error}"))
}

fn evaluate_through_omega(spec: &ProgramSpec, inputs: &[(&str, &[f32])], config: &RuntimeConfig) -> Result<Vec<TensorRecord>, String> {
    let (program, bound, outputs) = spec
        .plan_parts(inputs)
        .map_err(|error| format!("plan did not evaluate: {error}"))?;
    let named: Vec<(&str, QuantizedBlock<'_>)> = bound
        .iter()
        .map(|(name, values)| (*name, QuantizedBlock::Float32(values)))
        .collect();
    let (mut plan, _) = plan_named_with_config(config, &program, &[], &named, &outputs)
        .map_err(|error| format!("omega could not plan the program: {error}"))?;
    let evaluated = execute_plan_named(&mut plan, &named)
        .map_err(|error| format!("omega could not run the program: {error}"))?;
    spec.collect_outputs(&evaluated)
        .map_err(|error| format!("plan did not evaluate: {error}"))
}

pub fn mount(fallback: PipeHandle) -> PipeHandle {
    into_handle(
        RoutingPipe::new("proximad")
            .route(TENSOR_RUN_PATH, into_handle(TensorRunPipe::default()))
            .fallback(fallback),
    )
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::future::Future;
    use std::sync::atomic::Ordering;

    use proxima::{Bytes, Request, Response, SendPipe, into_handle};
    use proxima_cli::tensor_wire::{decode_response, encode_request};
    use proxima_tensor::spec::ProgramSpec;
    use proxima_tensor::test_support::Lcg;

    use omega::config::RuntimeConfig;

    use super::{SAFETENSORS_CONTENT_TYPE, TensorRunPipe, evaluate, evaluate_through_omega, mount};

    const PLAN_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../proxima-tensor/src/spec/fixtures/matmul_silu_softmax.toml"
    );
    const X_SHAPE: [u64; 2] = [2, 4];
    const W_SHAPE: [u64; 2] = [4, 3];

    fn plan_text() -> String {
        std::fs::read_to_string(PLAN_PATH).expect("matmul-silu-softmax fixture is readable")
    }

    fn matmul_inputs() -> (Vec<f32>, Vec<f32>) {
        let mut lcg = Lcg(0x5eed_0003);
        let x: Vec<f32> = (0..8).map(|_| lcg.next_unit() - 0.5).collect();
        let w: Vec<f32> = (0..12).map(|_| lcg.next_unit() - 0.5).collect();
        (x, w)
    }

    fn assert_outputs_agree(omega_outputs: &[(String, Vec<u64>, Vec<f32>)], cpu_outputs: &[(String, Vec<u64>, Vec<f32>)], tolerance: f32) {
        assert_eq!(omega_outputs.len(), cpu_outputs.len(), "one output per declared output");
        for ((omega_name, omega_shape, omega_values), (cpu_name, cpu_shape, cpu_values)) in omega_outputs.iter().zip(cpu_outputs) {
            assert_eq!(omega_name, cpu_name);
            assert_eq!(omega_shape, cpu_shape);
            for (found, expected) in omega_values.iter().zip(cpu_values) {
                assert!((found - expected).abs() <= tolerance, "omega {found} vs cpu {expected} for output {omega_name}");
            }
        }
    }

    #[test]
    fn omega_cpu_engine_agrees_with_the_cpu_evaluator_on_matmul_silu_softmax() {
        let spec: ProgramSpec = toml::from_str(&plan_text()).expect("plan parses");
        let (x, w) = matmul_inputs();
        let inputs: [(&str, &[f32]); 2] = [("x", &x), ("w", &w)];
        let config = RuntimeConfig::builder().with_engine("cpu").build().expect("cpu config validates");

        let through_omega = evaluate_through_omega(&spec, &inputs, &config).expect("omega runs the program");
        let direct = spec.evaluate_cpu(&inputs).expect("direct evaluation succeeds");

        assert_outputs_agree(&through_omega, &direct, 1e-6);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn omega_metal_engine_agrees_with_the_cpu_evaluator_on_matmul_silu_softmax() {
        let spec: ProgramSpec = toml::from_str(&plan_text()).expect("plan parses");
        let (x, w) = matmul_inputs();
        let inputs: [(&str, &[f32]); 2] = [("x", &x), ("w", &w)];
        let config = RuntimeConfig::builder()
            .with_engine("gpu")
            .with_gpu_driver("metal")
            .build()
            .expect("metal config validates");

        let routed = evaluate(&spec, &inputs, &config).expect("the gpu engine routes through omega");
        let direct = spec.evaluate_cpu(&inputs).expect("direct evaluation succeeds");

        assert_outputs_agree(&routed, &direct, 1e-4);
    }

    #[test]
    fn the_default_config_stays_on_the_cpu_evaluator() {
        let spec: ProgramSpec = toml::from_str(&plan_text()).expect("plan parses");
        let (x, w) = matmul_inputs();
        let inputs: [(&str, &[f32]); 2] = [("x", &x), ("w", &w)];

        let routed = evaluate(&spec, &inputs, &RuntimeConfig::default()).expect("default config evaluates");
        let direct = spec.evaluate_cpu(&inputs).expect("direct evaluation succeeds");

        assert_eq!(routed, direct, "an unset engine runs evaluate_cpu unchanged");
    }

    fn tensor_run_request(plan: &str, inputs: &[(&str, &[u64], &[f32])]) -> Request<Bytes> {
        let body = encode_request(plan, inputs).expect("request encodes");
        Request::builder()
            .method("POST")
            .path("/tensor/run")
            .body(body)
            .build()
            .expect("request builds")
    }

    fn matmul_request(plan: &str) -> Request<Bytes> {
        let (x, w) = matmul_inputs();
        tensor_run_request(plan, &[("x", &X_SHAPE, &x), ("w", &W_SHAPE, &w)])
    }

    #[proxima::test]
    async fn tensor_run_returns_softmax_bit_equal_to_in_process() {
        let plan = plan_text();
        let (x, w) = matmul_inputs();
        let pipe = TensorRunPipe::default();

        let response = pipe
            .call(matmul_request(&plan))
            .await
            .expect("pipe answers the request");

        assert_eq!(response.status, 200, "a valid plan is served");
        let content_type = response
            .metadata
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(b"content-type"))
            .map(|(_, value)| value.to_vec());
        assert_eq!(
            content_type.as_deref(),
            Some(SAFETENSORS_CONTENT_TYPE.as_bytes()),
            "the response declares the safetensors media type"
        );
        let decoded = decode_response(&response.payload).expect("response decodes");
        let spec: ProgramSpec = toml::from_str(&plan).expect("plan parses");
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

    #[proxima::test]
    async fn tensor_run_counts_requests_served() {
        let plan = plan_text();
        let pipe = TensorRunPipe::default();

        let first = pipe
            .call(matmul_request(&plan))
            .await
            .expect("first request answers");
        let second = pipe
            .call(matmul_request(&plan))
            .await
            .expect("second request answers");

        assert_eq!(first.status, 200, "first valid request is served");
        assert_eq!(second.status, 200, "second valid request is served");
        assert_eq!(
            pipe.served.load(Ordering::Relaxed),
            2,
            "two served requests are counted"
        );
    }

    #[proxima::test]
    async fn tensor_run_rejects_version_two_plan_with_400() {
        let plan = plan_text().replace("version = 1", "version = 2");
        assert!(
            plan.contains("version = 2"),
            "the fixture carries the version line this test rewrites"
        );
        let pipe = TensorRunPipe::default();

        let response = pipe
            .call(matmul_request(&plan))
            .await
            .expect("pipe answers the request");

        assert_eq!(response.status, 400, "an unsupported plan version is a 400");
        let message = String::from_utf8_lossy(&response.payload).into_owned();
        assert!(
            message.contains("unsupported plan version 2"),
            "the 400 body names the rejected version: {message}"
        );
        assert_eq!(
            pipe.served.load(Ordering::Relaxed),
            0,
            "a rejected request is not counted as served"
        );
    }

    #[proxima::test]
    async fn tensor_run_rejects_missing_input_with_400() {
        let plan = plan_text();
        let (x, _) = matmul_inputs();
        let pipe = TensorRunPipe::default();

        let response = pipe
            .call(tensor_run_request(&plan, &[("x", &X_SHAPE, &x)]))
            .await
            .expect("pipe answers the request");

        assert_eq!(response.status, 400, "a missing bind input is a 400");
        let message = String::from_utf8_lossy(&response.payload).into_owned();
        assert_eq!(
            message, "plan did not evaluate: no binding supplied for input `w`",
            "the 400 body names the missing input"
        );
    }

    struct ControlPlaneMarker;

    impl SendPipe for ControlPlaneMarker {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = proxima::ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, proxima::ProximaError>> + Send {
            async { Ok(Response::ok("control plane marker")) }
        }
    }

    #[proxima::test]
    async fn routing_sends_tensor_run_to_tensor_pipe_and_other_paths_to_fallback() {
        let plan = plan_text();
        let handle = mount(into_handle(ControlPlaneMarker));

        let tensor = handle
            .call(matmul_request(&plan))
            .await
            .expect("tensor route answers");
        let other = handle
            .call(
                Request::builder()
                    .method("GET")
                    .path("/pipelines")
                    .build()
                    .expect("request builds"),
            )
            .await
            .expect("fallback route answers");

        assert_eq!(tensor.status, 200, "/tensor/run reaches the tensor pipe");
        assert_eq!(
            other.payload.as_ref(),
            b"control plane marker",
            "every other path reaches the fallback"
        );
    }
}

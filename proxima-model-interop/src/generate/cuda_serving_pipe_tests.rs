#![cfg(all(test, feature = "cuda", feature = "interop-bgpool"))]

use super::chunked_prefill_tests::dense_checkpoint;
use super::cuda_serving_pipe::CudaServingPipe;
use super::{LoadedModel, ServingConfig};
use crate::cuda_async::CudaGpuService;
use crate::serving::{GPU_LAYERS_ALL, SpeculativeConfig};
use crate::{AsyncTokenProducer, Phase, TokenEvent};
use core::ops::ControlFlow;
use proxima_primitives::pipe::Pipe;

struct BreakAfterFirstToken {
    events: Vec<(u32, String, Phase, usize)>,
}

impl AsyncTokenProducer for BreakAfterFirstToken {
    async fn produce(&mut self, event: TokenEvent<'_>) -> ControlFlow<(), ()> {
        self.events.push((
            event.token_id,
            String::from(event.text_piece),
            event.phase,
            event.step,
        ));
        if event.phase == Phase::Token {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

#[proxima::test]
async fn test_omega_async_serving_pipe_awaits_shared_cuda_service() {
    let bytes = dense_checkpoint();
    let parsed = proxima_gguf::pipe::parse_complete(&bytes).expect("parse dense fixture");
    let model = LoadedModel::load(&parsed, &bytes).expect("load dense fixture");
    let sync_config = ServingConfig {
        gpu_layers: 0,
        prompt_cache: crate::serving::PromptCacheConfig::off(),
        speculative: SpeculativeConfig::none(),
        ..ServingConfig::default()
    };
    let sync_result = model
        .generate_with_serving_config("A", 2, sync_config)
        .expect("synchronous CPU incumbent completes");
    let service = CudaGpuService::start_fake().expect("start fake cuda owner");
    let config = ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        prompt_cache: crate::serving::PromptCacheConfig::off(),
        speculative: SpeculativeConfig::none(),
        ..ServingConfig::default()
    };
    let pipe = CudaServingPipe::new(&model, &service, config);
    let result = pipe.call((String::from("A"), 2)).await;
    let execution_count = service.fake_execute_count();
    let cpu_refusal = model
        .generate_async_with_serving_config(
            "A",
            2,
            ServingConfig {
                gpu_layers: 0,
                prompt_cache: crate::serving::PromptCacheConfig::off(),
                speculative: SpeculativeConfig::none(),
                ..ServingConfig::default()
            },
            &service,
        )
        .await;
    let fusion_refusal = model
        .generate_async_with_serving_config(
            "A",
            2,
            ServingConfig {
                cached_attention_fusion: false,
                ..config
            },
            &service,
        )
        .await;
    assert!(matches!(
        cpu_refusal,
        Err(crate::InteropError::AsyncCudaRequiresGpu)
    ));
    assert!(matches!(
        fusion_refusal,
        Err(crate::InteropError::AsyncCudaGenerationUnsupported {
            mode: "per-request fusion environment overrides"
        })
    ));
    assert_eq!(service.fake_execute_count(), execution_count);
    service.shutdown().await.expect("drain fake cuda owner");
    let (generated, text, stopped_by_eos) = result.expect("async serving pipe completes");
    assert_eq!((generated, text, stopped_by_eos), sync_result);
    assert_eq!(execution_count, 2);
}

#[proxima::test]
async fn test_omega_async_token_producer_preserves_break_and_event_order() {
    let bytes = dense_checkpoint();
    let parsed = proxima_gguf::pipe::parse_complete(&bytes).expect("parse dense fixture");
    let model = LoadedModel::load(&parsed, &bytes).expect("load dense fixture");
    let cpu_config = ServingConfig {
        gpu_layers: 0,
        prompt_cache: crate::serving::PromptCacheConfig::off(),
        speculative: SpeculativeConfig::none(),
        ..ServingConfig::default()
    };
    let mut sync_events = Vec::new();
    let sync_result = model
        .generate_streaming("A", 5, cpu_config, &mut |event| {
            sync_events.push((
                event.token_id,
                String::from(event.text_piece),
                event.phase,
                event.step,
            ));
            if event.phase == Phase::Token {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .expect("synchronous producer stops on the first token event");
    let service = CudaGpuService::start_fake().expect("start fake cuda owner");
    let gpu_config = ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        ..cpu_config
    };
    let mut async_producer = BreakAfterFirstToken { events: Vec::new() };
    let async_result = model
        .generate_async_streaming_with_serving_config(
            "A",
            5,
            gpu_config,
            &service,
            &mut async_producer,
        )
        .await;
    service.shutdown().await.expect("drain fake cuda owner");
    assert_eq!(
        async_result.expect("async event producer completes"),
        sync_result
    );
    assert_eq!(async_producer.events, sync_events);
}

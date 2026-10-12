#![cfg(all(feature = "instrument", target_os = "macos"))]

use omega::metal::ResidentContext;
use omega::{execute_plan, plan, set_capture_step, take_captured_dispatches};
use proxima_tensor::{
    DType, Extent, IndexMap, NumericPolicy, Op, QuantizedBlock, ScalarOp, append, projection,
};

#[test]
fn resident_context_keeps_real_dispatch_captures_with_their_owner() {
    temp_env::with_vars(
        [
            ("PROXIMA_CAPTURE_LIVE", Some("1")),
            ("PROXIMA_CAPTURE_NODES", Some("all")),
            ("PROXIMA_CAPTURE_STEPS", None),
            ("PROXIMA_CAPTURE_DUMP_DIR", None),
        ],
        || {
            let mut program = Vec::new();
            let input = append(
                &mut program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: vec![Extent::Static(2)],
                    name: None,
                },
            );
            let output = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Reciprocal,
                    operands: vec![(input, IndexMap::Affine(projection(1, &[0])))],
                    name: None,
                },
            );
            let target_values = [1.0, 2.0];
            let draft_values = [4.0, 8.0];
            let target_blocks = [QuantizedBlock::Float32(&target_values)];
            let draft_blocks = [QuantizedBlock::Float32(&draft_values)];
            let execution = plan(
                &program,
                &[],
                &target_blocks,
                &[output],
                NumericPolicy::default(),
            )
            .unwrap_or_else(|error| panic!("plan reciprocal capture fixture: {error}"));
            let mut target = ResidentContext::default();
            let mut draft = ResidentContext::default();

            {
                let _target_active = target.activate();
                assert!(take_captured_dispatches().is_empty());
                set_capture_step(11);
                let evaluated = execute_plan(&execution, &target_blocks)
                    .unwrap_or_else(|error| panic!("execute target capture fixture: {error}"));
                let (values, _) = evaluated
                    .get(output)
                    .unwrap_or_else(|| panic!("target output is retained"));
                assert_eq!(values, &[1.0, 0.5]);

                {
                    let _draft_active = draft.activate();
                    assert!(take_captured_dispatches().is_empty());
                    set_capture_step(22);
                    let evaluated = execute_plan(&execution, &draft_blocks)
                        .unwrap_or_else(|error| panic!("execute draft capture fixture: {error}"));
                    let (values, _) = evaluated
                        .get(output)
                        .unwrap_or_else(|| panic!("draft output is retained"));
                    assert_eq!(values, &[0.25, 0.125]);
                }

                let captures = take_captured_dispatches();
                assert_eq!(captures.len(), 1);
                assert_eq!(captures[0].step, 11);
                assert_eq!(captures[0].node, output.0);
            }

            {
                let mut draft_active = draft.activate();
                let captures = take_captured_dispatches();
                assert_eq!(captures.len(), 1);
                assert_eq!(captures[0].step, 22);
                assert_eq!(captures[0].node, output.0);
                drop(captures);
                set_capture_step(23);
                execute_plan(&execution, &draft_blocks)
                    .unwrap_or_else(|error| panic!("replay draft capture fixture: {error}"));
                draft_active.clear();
                assert!(take_captured_dispatches().is_empty());
            }
            set_capture_step(0);
        },
    );
}

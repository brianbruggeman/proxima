//! The pinned MLIR tools: `mlir-opt` and `mlir-runner` 21.1.8 from Homebrew
//! LLVM, the runner libraries, and loud failure when either is absent or wrong.

#[path = "support/tool.rs"]
mod tool;

mod mlir_tool_cases {
    use std::path::Path;

    use super::tool::{
        PINNED_VERSION, Tool, ToolError, check_version, lower_to_llvm, mlir_opt, mlir_runner,
        parse_printed_memref, parse_version, run_main,
    };

    const MATMUL_PROBE: &str = r"
func.func private @printMemrefF32(tensor<*xf32>)

func.func @main() {
  %lhs = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %rhs = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x2xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
  %product = linalg.matmul ins(%lhs, %rhs : tensor<2x2xf32>, tensor<2x2xf32>) outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
  %erased = tensor.cast %product : tensor<2x2xf32> to tensor<*xf32>
  call @printMemrefF32(%erased) : (tensor<*xf32>) -> ()
  return
}
";

    #[test]
    fn mlir_opt_reports_llvm_version_21_1_8() -> Result<(), ToolError> {
        let opt = mlir_opt()?;

        let banner = opt.run(["--version"], "")?;

        assert_eq!(parse_version(&banner), Some("21.1.8"));
        assert_eq!(PINNED_VERSION, "21.1.8");
        Ok(())
    }

    #[test]
    fn mlir_runner_resolves_and_executes_the_matmul_probe_printing_19_22_43_50()
    -> Result<(), ToolError> {
        let opt = mlir_opt()?;
        let runner = mlir_runner()?;

        let lowered = lower_to_llvm(&opt, MATMUL_PROBE)?;
        let printed = run_main(&runner, &lowered)?;

        assert_eq!(parse_printed_memref(&printed)?, [19.0, 22.0, 43.0, 50.0]);
        Ok(())
    }

    #[test]
    fn runner_libraries_libmlir_c_runner_utils_and_libmlir_runner_utils_exist()
    -> Result<(), ToolError> {
        let runner = mlir_runner()?;

        let libraries = runner.runner_libraries()?;

        let names: Vec<_> = libraries
            .iter()
            .filter_map(|path| path.file_name()?.to_str())
            .collect();
        assert_eq!(
            names,
            ["libmlir_runner_utils.dylib", "libmlir_c_runner_utils.dylib"]
        );
        assert!(libraries.iter().all(|path| path.is_file()));
        Ok(())
    }

    #[test]
    fn missing_tool_error_names_mlir_opt() {
        let absent = Tool::at("mlir-opt", "/nonexistent/mlir-opt");

        let Err(error) = absent.pinned() else {
            panic!("a path that does not exist must not pin");
        };

        assert!(matches!(
            error,
            ToolError::Missing {
                tool: "mlir-opt",
                ..
            }
        ));
        let message = error.to_string();
        assert!(message.contains("mlir-opt"), "message was {message}");
        assert!(
            message.contains("/nonexistent/mlir-opt"),
            "message was {message}"
        );
    }

    #[test]
    fn wrong_version_line_20_0_0_is_rejected() {
        let banner = "Homebrew LLVM version 20.0.0\n  Optimized build.\n";

        let Err(error) = check_version("mlir-opt", Path::new("/opt/old/mlir-opt"), banner) else {
            panic!("a 20.0.0 banner must not satisfy the 21.1.8 pin");
        };

        assert!(matches!(
            error,
            ToolError::WrongVersion { ref found, expected: "21.1.8", .. } if found == "20.0.0"
        ));
        assert!(error.to_string().contains("20.0.0"));
    }
}

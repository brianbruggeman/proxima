use alloc::format;
use alloc::vec::Vec;

use super::dialect::Cuda;
use crate::cuda::target::Target;
use crate::msl::dialect::Msl;
use crate::render::{self, Dialect, HardwareScalar};

const TARGETS: [Target; 2] = [Target::CUDA, Target::HIP];

fn wide(dialect: &Cuda) -> HardwareScalar {
    HardwareScalar::widest(dialect).expect("the cuda dialect spells a wide scalar")
}

fn narrower_rows(dialect: &Cuda) -> Vec<HardwareScalar> {
    let wide = wide(dialect);
    HardwareScalar::all(dialect)
        .filter(|row| *row != wide)
        .collect()
}

#[test]
fn cuda_spells_exactly_the_hardware_rows_metal_spells() {
    for target in TARGETS {
        let cuda = Cuda { target };

        let cuda_rows: Vec<usize> = HardwareScalar::all(&cuda)
            .map(HardwareScalar::index)
            .collect();
        let metal_rows: Vec<usize> = HardwareScalar::all(&Msl)
            .map(HardwareScalar::index)
            .collect();

        assert!(!cuda_rows.is_empty(), "the cuda dialect spells no row");
        assert_eq!(cuda_rows, metal_rows, "{}", target.language_name());
    }
}

#[test]
fn every_cuda_row_has_a_distinct_type_and_both_bit_reinterpretations() {
    for target in TARGETS {
        let cuda = Cuda { target };
        let rows: Vec<HardwareScalar> = HardwareScalar::all(&cuda).collect();
        let mut names: Vec<_> = rows.iter().filter_map(|row| cuda.scalar(*row)).collect();

        for row in &rows {
            assert!(cuda.to_bits(*row, "value").is_some(), "{row:?}");
            assert!(cuda.to_scalar(*row, "code").is_some(), "{row:?}");
        }
        names.sort();
        names.dedup();
        assert_eq!(names.len(), rows.len(), "{names:?}");
    }
}

#[test]
fn the_bfloat_row_is_named_by_the_target() {
    for target in TARGETS {
        let cuda = Cuda { target };

        let names: Vec<_> = narrower_rows(&cuda)
            .into_iter()
            .filter_map(|row| cuda.scalar(row))
            .collect();

        assert_eq!(names.len(), 2, "{names:?}");
        assert!(
            names.iter().any(|name| *name == target.bfloat_type),
            "{names:?}"
        );
    }
}

#[test]
fn a_narrowing_hardware_store_renders_the_cuda_conversion_through_the_layout_row() {
    for target in TARGETS {
        let cuda = Cuda { target };
        let wide = wide(&cuda);
        let rows = narrower_rows(&cuda);

        assert!(!rows.is_empty(), "no narrower row to store");
        for row in rows {
            let name = cuda.scalar(row).expect("a spelled row has a name");
            let reinterpret = if name == target.bfloat_type {
                "__bfloat16_as_ushort"
            } else {
                "__half_as_ushort"
            };

            let store = render::store(&cuda, wide, row.layout(), "probe_store")
                .expect("a hardware row narrows through the neutral render");

            assert!(
                store.contains(&format!(
                    "return (unsigned int){reinterpret}(({name})value);"
                )),
                "{name}:\n{store}"
            );
            assert!(
                store.contains("__device__ __forceinline__ unsigned int probe_store("),
                "{store}"
            );
            assert!(
                store.contains("__float_as_uint(value)"),
                "nan branch reads the wide bits:\n{store}"
            );
        }
    }
}

#[test]
fn a_hardware_load_reinterprets_the_code_then_widens_through_the_layout_row() {
    for target in TARGETS {
        let cuda = Cuda { target };
        let wide = wide(&cuda);
        let wide_name = cuda.scalar(wide).expect("the wide row has a name");
        let rows = narrower_rows(&cuda);

        assert!(!rows.is_empty(), "no narrower row to load");
        for row in rows {
            let name = cuda.scalar(row).expect("a spelled row has a name");
            let reinterpret = if name == target.bfloat_type {
                "__ushort_as_bfloat16"
            } else {
                "__ushort_as_half"
            };

            let load = render::load(&cuda, wide, row.layout(), "probe_load")
                .expect("a hardware row loads through the neutral render");

            assert!(load.contains(reinterpret), "{name}:\n{load}");
            assert!(
                load.contains(&format!("{wide_name} probe_load(unsigned int code)")),
                "{load}"
            );
        }
    }
}

#[test]
fn the_cuda_integers_are_the_card_c_types_of_their_width() {
    let cuda = Cuda {
        target: Target::CUDA,
    };

    assert_eq!(cuda.unsigned(4).as_deref(), Some("unsigned int"));
    assert_eq!(cuda.signed(8).as_deref(), Some("long long"));
    assert_eq!(cuda.unsigned(3), None);
}

//! Byte-for-byte oracle for the block encoder: the bytes every layout this crate encodes produced
//! before the quantizer moved into `proxima_numeric`, captured from the implementation it replaced.
//!
//! The input is a deterministic splitmix stream, identical on both sides, so a disagreement is a
//! disagreement about the encoder and never about the data. A layout that stops encoding, or that
//! starts producing different bytes, fails here rather than downstream in a quantized checkpoint.

use proxima_gguf::quant::block_layout_encode::{quantize, reach};
use proxima_numeric::NumericFormat;

/// Every layout the pre-move encoder accepted, with the bytes it produced for [`values`].
///
/// Pinned by count: a new encodable layout must be added here, so the oracle cannot silently stop
/// covering one. The rows are in [`NumericFormat::ALL`] order.
const ORACLE: [(&str, &[&str]); 10] = [
    (
        "q8_0",
        &[
            "6932fbf11d19055fa211dd3b0f255701fce8d300f2ea7f0df631bc0b092804f6ed42b12f09394feccaa7e8566d60dffe",
            "3df1d658f1e929a6c48c35fdeb20d5a4f88193f1ec32e0f2db27cb7f101b302b1cc646030af8b9b956410808d733be35",
            "9e05141d35fd",
        ],
    ),
    (
        "q4_0",
        &[
            "794388789a772b959887593927fc8c7a686af8422381a188799a46437896b8c676e63791aa4376a96aa58b9e68497684",
            "937d8159794b",
        ],
    ),
    (
        "q5_0",
        &[
            "453f9b6693f02104ee92f6fcd6509ea3b636032d415ff93ca855bb728b1cb152bb7067f2f72a819ea4dacef7743f5bf7",
            "4dcf71f0a924f7e730af5667f70cf0d272a2",
        ],
    ),
    (
        "q8_1",
        &[
            "e23269cc0bdbef0126f0bcf61dff00000caf00dff528d650a308dee84413d57fffdd33eccd32cc57f66b0a0f140713d4",
            "5812fa071cbcd0ff7d1325d543db22f2fbdb3021520d7ffc1f32bc57ee2f4f0959d910f0fd3d1569f110063eca451f7f",
            "fc113ab7ce752c0ffcddfce5",
        ],
    ),
    (
        "q8_k",
        &[
            "309d3f3e0c0ee4f2215fabf721093fb6aecf04e72233488bba05fe09e0b664950426efbe1a33e1fb1ccbba310f18f8ed",
            "e2f012262dfa1ce7dc376d13feddfb1ad820520d29ccd9fa6034fa0e2f12393248f21039e0125ad115fbe6194d16c307",
            "f712d71ae9fcf51517c12baa8ffa26cffdbf92d4f72a1edfa7f0c4151dfbcdceacac11f5e5b9e64faf06230220bc6876",
            "04eb36472763e3ef00e3cdddec36041bc32f5b7fbe0eee1db0b8d209fbe325e147de3ef50ef9ff019d45110f06fc075b",
            "4a05e7ecd6adbdf210d531099bb120eb0d24ca00d6cef5fbc01deffab7fbfd3bf5b1588e0a12d52ca23390fe0ccae6a7",
            "3433f107f7511dac8303dc1939dbf2780026e2f700000000000000000000000000000000000000000000000000000000",
            "000000003f7f503ed7a99006f8cdcff00313fd1c14d02e6ad20814ec38ab6a521114e94ec1603ceee5fa0b24d4d4f425",
            "1119ccfe5deff1d708c3e0ecfd1213b40ce2a1fb1dfab60d5ba5ffde19be534d2b03f719373f177e11fbf3bdf86e5311",
            "112707d994f703130bddc39f62030ae345bb182f0cfc2bb245fed604ff28c01b41da0e22f7f94be4eefc081ada3196f9",
            "f0fb25010df385fefdfa205e903dfa14fd24c0c8abf0dbd609b7318408133b4b2bf0da06f5ce294b6d130c0af4c40076",
            "3df918fb4cc95004f7e8dcfbc81b2c0526d344f60e6e0904f5d1ca457c08eccb01324a79f821d61d48127ffa09f1bf07",
            "e591f0e5ea13ee269fefec152c9fd670f60cf3d11fb3d511000000000000000000000000000000000000000000000000",
            "0000000000000000cc8f573e23db315bf6350e00d9b6cee44af8f705edd9b4c01025282e062ead09e4dbc2e056910bfe",
            "0cbb416064fd0bf900554867f31f2bd434f0b2f3f6e922f327b809e8292c054b81070939044af458ede801e23c71f2f5",
            "ff130b1ac9a0f3faca33301ac8f5e2c942bf1e2113e5ca06d33706f2190043f40ce607fd0f3aea317005ea14df35e08d",
            "0b0a33c1164c7d121b2dfd55c0580ce703bef716c50c1c07de2fb755fd03322b02ca340301cdc3bbdddaf4041cf2b41d",
            "93f7df14283761ef0015d8cb23c7d3fb1d08f71898f5f9f53439e8c7d80714fef2ccc0fe1109041fe34b0ef8e801d952",
            "2ed8fef72d06fff2e2030e2db52e5e450824cb291b95b7ef0ed445340000000000000000000000000000000000000000",
            "000000000000000000000000",
        ],
    ),
    (
        "tq2_0",
        &[
            "5591554565149499955145555694959559454656559511515545551215915155655655651a5515556545545455595955",
            "65455495595545651594511159456554bb4e555594599555655585569155515545555555515966665555515555555565",
            "54995615195555555554551559555666541591516545655555555545645556195559e44e5155655565561995555656a4",
            "565555556555555595515555555555559159655545455454195151555545949519595555555611955545556454555545",
            "55565656b74e",
        ],
    ),
    (
        "q1_0",
        &[
            "b54e16fbe51830052caeb6e82a06a041366fea4e5ec919208d8bc114e09a102fe1f1c311cc4e8daa34c3467b1c437324",
            "2d51c0830529",
        ],
    ),
    (
        "q2_0",
        &[
            "8a4e55559556555481554015551554159549b54e55598154551555155155561555496554ea4e95565595411555155691",
            "554055545556",
        ],
    ),
    (
        "pq2_0",
        &[
            "de4e5549555655565555555555555551555155595556a05544555965541955655558d24e155625555655515155555548",
            "555555569655545555555595555a555545564555ed4e5895505555515545655055549855655552955665555955599955",
            "455564555505",
        ],
    ),
    (
        "q8_1_grouped",
        &[
            "b9315dcd193105cde7316e55fa329d4d04d608265545a200d73bdddba8901620e9eed77d811409da362810f7060e0de3",
            "1dd50e0e2ac5f9a1b08d08f6da193a7faef5d231faec425d0cd3c92b0bd837ecdad039443670f5102801c4677f001a3d",
            "046c6d93f0d6e8d41148b5f5e9edbb550f81ef1df522f004f2090a351caa52d4f0f8343e4d47bd052333f03aeb8c0ae9",
            "e2301fd0813281ba0a322e4e16321ad405c8ba786b131ccda9cbe9fcf62effdbffcc0dfcdada1fc77f100228e3c3009e",
            "10fc1f2bd95df8820604edb2e2fcbd07040b20dcc67ffd22ed37a13c7412ddd8eda63ad50f17dbf715f27fef0929def4",
            "597313e5ee55f0db38f40b16c1c3d5f81007ffb84fa3ad15dc00f5548555f603d3f9011d81fb2738b6f10a19fdf3c1ed",
            "d432d4b6bc3177cde931dbd24632e44f57aadaf1e6222a24e3950b0af3c212518106f6f641d652760fe4231ed6bd25f9",
            "02da26b081c50be91f089adf0c0ed440004d44c2f8f5d7502a1c31fd070bcb1481c90d173bba15555afcdf3aed01b405",
            "ffded4fde8950f1520efc365d89bf605042108343a04213437e6db1b13f30f08d85e81030dfb49c0a8dff80bcf4d36d6",
        ],
    ),
];

fn values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|index| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut mixed = state;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            mixed ^= mixed >> 31;
            let unit = (mixed >> 11) as f32 / (1u64 << 53) as f32;
            let signed = unit * 2.0 - 1.0;
            4.0 * signed * (1.0 + (index % 7) as f32)
        })
        .collect()
}

/// Formats the pre-move encoder refused and this one writes. Pinned so a format that starts
/// encoding cannot slip in without a reviewer seeing it; the bytes for these are pinned by
/// `proxima-numeric`'s own hand-computed OCP MX vectors instead of by a pre-move capture.
const NEWLY_ENCODABLE: [&str; 5] = ["mxfp4", "nvfp4", "tq1_0", "fp6_e3m2", "ptq1_0"];

#[test]
fn every_encodable_layout_reproduces_the_pre_move_bytes() {
    let mut encodable: Vec<&str> = Vec::new();
    for format in NumericFormat::ALL {
        let Some(layout) = format.block_layout() else {
            continue;
        };
        if reach(&layout).is_err() {
            continue;
        }
        let name = format.name().expect("every format is named");
        let input = values(
            usize::from(layout.elements) * 3,
            u64::from(layout.block_bytes) + 17,
        );
        let blocks = input.len() / usize::from(layout.elements);
        let mut packed = vec![0u8; blocks * usize::from(layout.block_bytes)];
        quantize(&layout, &input, &mut packed).expect("a supported layout encodes");
        let hexed: String = packed.iter().map(|byte| format!("{byte:02x}")).collect();
        match ORACLE.iter().find(|(pinned, _)| *pinned == name) {
            Some((_, pinned)) => assert_eq!(hexed, pinned.concat(), "{name} bytes"),
            None => assert!(
                NEWLY_ENCODABLE.contains(&name),
                "{name} encodes and appears in neither the pre-move oracle nor the newly-encodable list"
            ),
        }
        encodable.push(name);
    }
    let captured: Vec<&str> = ORACLE.iter().map(|(name, _)| *name).collect();
    for name in &captured {
        assert!(
            encodable.contains(name),
            "the pre-move encoder wrote {name} and this one does not"
        );
    }
    assert_eq!(
        encodable.len(),
        captured.len() + NEWLY_ENCODABLE.len(),
        "pinned: every encodable layout is accounted for, old or new"
    );
}

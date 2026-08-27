//! M5b's DTB half: `dtc` is the devicetree-compiler reference
//! implementation and, per principle 14, the incumbent whose output this
//! test treats as ground truth. It decompiles the blob
//! [`proxima_vm::dtb::build_minimal_aarch64_boot_dtb`] produces
//! (`dtc -I dtb -O dts`) and asserts the decompiled source names every
//! node and property the builder wrote — divergence would mean our tree
//! shape disagrees with what a real `dtc`-consuming toolchain (U-Boot,
//! the Linux kernel's own `dtc`-built `.dtb` verification) would accept.
//!
//! `which dtc` gates this test at runtime rather than at compile time:
//! recording UNMEASURED-dtc-not-installed and skipping is the honest
//! answer on a host without the devicetree compiler, not a hard failure.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::process::Command;

use proxima_vm::dtb::{BootParams, QemuVirtLayout, build_minimal_aarch64_boot_dtb};

fn dtc_path() -> Option<String> {
    let output = Command::new("which").arg("dtc").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8(output.stdout).ok()?;
    Some(path.trim().to_owned())
}

#[test]
fn dtc_decompiles_the_built_blob_into_every_node_and_property_this_slice_writes() {
    let Some(dtc) = dtc_path() else {
        eprintln!("UNMEASURED-dtc-not-installed: skipping dtc differential");
        return;
    };

    let boot = BootParams {
        ram_base: 0x4000_0000,
        ram_size: 0x4000_0000,
        bootargs: "console=hvc0 root=/dev/vda rw",
        initrd: Some((0x4800_0000, 0x4900_0000)),
    };
    let blob = build_minimal_aarch64_boot_dtb(&QemuVirtLayout::single_vcpu(), &boot)
        .expect("worked-example layout must build");

    let directory = tempfile::tempdir().expect("create tempdir for the dtb under test");
    let dtb_path = directory.path().join("boot.dtb");
    std::fs::write(&dtb_path, &blob).expect("write the built blob to disk for dtc");

    let output = Command::new(dtc)
        .args(["-I", "dtb", "-O", "dts"])
        .arg(&dtb_path)
        .output()
        .expect("run dtc against the built blob");
    assert!(
        output.status.success(),
        "dtc rejected our blob: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let decompiled = String::from_utf8(output.stdout).expect("dtc emits UTF-8 dts text");

    for expected in [
        r#"compatible = "linux,dummy-virt";"#,
        "#address-cells = <0x02>;",
        "#size-cells = <0x02>;",
        "interrupt-parent = <0x01>;",
        "device_type = \"memory\";",
        "reg = <0x00 0x40000000 0x00 0x40000000>;",
        r#"bootargs = "console=hvc0 root=/dev/vda rw";"#,
        "linux,initrd-start = <0x00 0x48000000>;",
        "linux,initrd-end = <0x00 0x49000000>;",
        "cpu@0 {",
        r#"compatible = "arm,cortex-a72";"#,
        r#"enable-method = "psci";"#,
        "psci {",
        r#"compatible = "arm,psci-0.2";"#,
        r#"method = "hvc";"#,
        "interrupt-controller@8000000 {",
        r#"compatible = "arm,gic-v3";"#,
        "reg = <0x00 0x8000000 0x00 0x10000 0x00 0x80a0000 0x00 0x20000>;",
        "phandle = <0x01>;",
        "timer {",
        r#"compatible = "arm,armv8-timer";"#,
        "interrupts = <0x01 0x0d 0x04 0x01 0x0e 0x04 0x01 0x0b 0x04 0x01 0x0a 0x04>;",
        "always-on;",
        "pl011@9000000 {",
        r#"compatible = "arm,pl011", "arm,primecell";"#,
        "reg = <0x00 0x9000000 0x00 0x1000>;",
        "interrupts = <0x00 0x01 0x04>;",
        r#"clock-names = "uartclk", "apb_pclk";"#,
        "stdout-path = \"/pl011@9000000\";",
        "apb-pclk {",
        r#"compatible = "fixed-clock";"#,
        "clock-frequency = <0x16e3600>;",
        r#"clock-output-names = "clk24mhz";"#,
    ] {
        assert!(
            decompiled.contains(expected),
            "dtc's decompiled tree is missing `{expected}`:\n{decompiled}"
        );
    }
}

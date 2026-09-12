//! Probe the portable GPU adapter without loading a model.

fn main() {
    match omega::probe_wgpu() {
        Ok(info) => println!(
            "wgpu_smoke: name={:?} backend={} device_type={} max_buffer_size={} max_storage_buffer_binding_size={}",
            info.name,
            info.backend,
            info.device_type,
            info.max_buffer_size,
            info.max_storage_buffer_binding_size,
        ),
        Err(error) => {
            eprintln!("wgpu_smoke: unavailable: {error}");
            std::process::exit(0);
        }
    }
}

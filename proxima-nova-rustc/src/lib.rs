//! proxima-nova codegen backend: loaded by `rustc -Zcodegen-backend`.
//!
//! `codegen_crate` lowers every fn mono item's MIR to a nova `Function` (see [`mir`]), runs
//! `proxima_nova::pipeline::compile()` on prime, writes the Mach-O object to the cgu temp path and
//! hands it to rustc's own `link_binary`. `-Cllvm-args=nova-list` keeps the slice-5 listing mode;
//! `-Cllvm-args=nova-layout` also prints one `nova_layout` line per struct local and still links;
//! `-Cllvm-args=nova-ir` prints one `nova_ir` line per lowered function and still links.

#![feature(rustc_private)]

extern crate rustc_abi;
extern crate rustc_attr_ir;
extern crate rustc_codegen_ssa;
extern crate rustc_driver as _;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_public;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_structures;

use std::any::Any;
use std::collections::{HashSet, VecDeque};
use std::fs;
use std::future::Future;
use std::thread;
use std::time::{Duration, Instant};

use conflaguration::Settings;
use futures::FutureExt;
use futures::executor::block_on;
use mir_error::MirError;
use prime::os::background::ProximaBackgroundPool;
use proxima_nova::ir::Function;
use proxima_nova::object::{ObjectBuilder, ObjectError};
use proxima_nova::pipeline::{Encoded, backend};
use proxima_primitives::pipe::Pipe;
use rustc_codegen_ssa::back::archive::ArArchiveBuilderBuilder;
use rustc_codegen_ssa::back::link::link_binary;
use rustc_codegen_ssa::back::write::produce_final_output_artifacts;
use rustc_codegen_ssa::target_features::internal_target_features;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_codegen_ssa::{CompiledModule, CompiledModules, CrateInfo, ModuleKind, TargetConfig};
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph::WorkProductMap;
use rustc_middle::mir::interpret::AllocId;
use rustc_middle::mono::MonoItem as InternalMonoItem;
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_public::rustc_internal;
use rustc_session::config::{OutputFilenames, OutputType};
use rustc_session::{EarlySession, IncrCompSession, Session};
use rustc_span::def_id::DefId;

mod data;
mod mir;
mod mir_error;

const LIST_MODE: &str = "nova-list";
const LAYOUT_MODE: &str = "nova-layout";
const IR_MODE: &str = "nova-ir";
const TIME_MODE: &str = "nova-time";

struct NovaBackend;

fn layout_mode(sess: &Session) -> bool {
    sess.opts.cg.llvm_args.iter().any(|arg| arg == LAYOUT_MODE)
}

fn ir_mode(sess: &Session) -> bool {
    sess.opts.cg.llvm_args.iter().any(|arg| arg == IR_MODE)
}

fn time_mode(sess: &Session) -> bool {
    sess.opts.cg.llvm_args.iter().any(|arg| arg == TIME_MODE)
}

fn list_mode(sess: &Session) -> bool {
    sess.opts.cg.llvm_args.iter().any(|arg| arg == LIST_MODE)
}

fn print_mono_items(tcx: TyCtxt<'_>) {
    let (_, instances, _) = mono_instances(tcx);
    let ran = rustc_internal::run(tcx, || {
        for instance in &instances {
            let stable = rustc_internal::stable(*instance);
            println!("mono_item={}", stable.name());
        }
    });
    if ran.is_err() {
        tcx.dcx().fatal("proxima-nova: rustc_public refused to run");
    }
}

fn mono_instances(tcx: TyCtxt<'_>) -> (String, Vec<Instance<'_>>, Vec<DefId>) {
    let partitions = tcx.collect_and_partition_mono_items(());
    let name = partitions
        .codegen_units
        .first()
        .map_or_else(|| "nova".to_string(), |unit| unit.name().to_string());
    let items: Vec<InternalMonoItem<'_>> = partitions
        .codegen_units
        .iter()
        .flat_map(|unit| unit.items_in_deterministic_order(tcx))
        .map(|(item, _data)| item)
        .collect();
    let instances = items
        .iter()
        .filter_map(|item| match item {
            InternalMonoItem::Fn(instance) => Some(*instance),
            _ => None,
        })
        .collect();
    let statics = items
        .iter()
        .filter_map(|item| match item {
            InternalMonoItem::Static(def_id) => Some(*def_id),
            _ => None,
        })
        .collect();
    (name, instances, statics)
}

// statics first, then every memory allocation they or the functions named, each once
fn feed_data(
    tcx: TyCtxt<'_>,
    builder: &mut ObjectBuilder,
    statics: &[DefId],
    mut pending: Vec<AllocId>,
) {
    let mut objects = Vec::new();
    for def_id in statics {
        let object = data::static_object(tcx, *def_id, &mut pending)
            .unwrap_or_else(|error| refuse(tcx, &data::static_symbol(tcx, *def_id), error));
        objects.push(object);
    }
    let mut seen = HashSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        let object = data::memory_object(tcx, id, &mut pending)
            .unwrap_or_else(|error| refuse(tcx, &data::memory_symbol(id), error));
        objects.push(object);
    }
    for object in objects {
        let symbol = object.symbol.clone();
        if let Err(error) = builder.feed_data(object) {
            tcx.dcx().fatal(format!("proxima-nova: {symbol}: {error}"));
        }
    }
}

// the packing rustc_codegen_ssa back/metadata.rs:421-424 hands the object writer; the version is
// `sess.apple_deployment_target()` (rustc_session session.rs:384), MACOSX_DEPLOYMENT_TARGET or the 11.0 floor
fn packed_deployment_target(tcx: TyCtxt<'_>) -> u32 {
    let version = tcx.sess.apple_deployment_target();
    (u32::from(version.major) << 16) | (u32::from(version.minor) << 8) | u32::from(version.patch)
}

/// Streaming knobs, read from `NOVA_WORKERS` and `NOVA_WINDOW_PER_WORKER`.
#[derive(Debug, Clone, PartialEq, Eq, Settings)]
#[settings(prefix = "NOVA")]
struct StreamSettings {
    /// pool threads running the backend; 0 means one fewer than the available cores, at least 1
    #[setting(default = 0)]
    workers: usize,
    /// jobs in flight per worker; the window is `workers * window_per_worker`
    #[setting(default = 2)]
    window_per_worker: usize,
}

impl StreamSettings {
    fn resolved_workers(&self) -> usize {
        match self.workers {
            0 => thread::available_parallelism()
                .map_or(1, |cores| cores.get().saturating_sub(1).max(1)),
            fixed => fixed,
        }
    }

    fn window(&self) -> usize {
        self.resolved_workers() * self.window_per_worker.max(1)
    }
}

type JobResult = Result<Result<Encoded, ObjectError>, String>;

// a backend refusal moves Streaming to Faulted: dispatch stops, lowering goes on, so a later MirError still wins
enum Drive<Handle> {
    Streaming {
        window: VecDeque<Handle>,
        // boxed: the builder is 320 bytes against Faulted's 80, one allocation per crate
        builder: Box<ObjectBuilder>,
    },
    Faulted {
        first: ObjectError,
    },
}

impl<Handle: Future<Output = JobResult>> Drive<Handle> {
    fn new(minos: u32) -> Self {
        Self::Streaming {
            window: VecDeque::new(),
            builder: Box::new(ObjectBuilder::new(minos)),
        }
    }

    async fn await_oldest(self, tcx: TyCtxt<'_>) -> Self {
        let Self::Streaming {
            mut window,
            mut builder,
        } = self
        else {
            return self;
        };
        let Some(handle) = window.pop_front() else {
            return Self::Streaming { window, builder };
        };
        match handle.await {
            Err(error) => tcx
                .dcx()
                .fatal(format!("proxima-nova: backend job failed: {error}")),
            Ok(Err(first)) => Self::Faulted { first },
            Ok(Ok(encoded)) => match builder.feed(encoded) {
                Ok(()) => Self::Streaming { window, builder },
                Err(first) => Self::Faulted { first },
            },
        }
    }

    async fn submit(
        mut self,
        tcx: TyCtxt<'_>,
        capacity: usize,
        spawn: impl FnOnce() -> Handle,
    ) -> Self {
        if matches!(&self, Self::Streaming { window, .. } if window.len() >= capacity) {
            self = self.await_oldest(tcx).await;
        }
        match self {
            Self::Streaming {
                mut window,
                builder,
            } => {
                window.push_back(spawn());
                Self::Streaming { window, builder }
            }
            faulted @ Self::Faulted { .. } => faulted,
        }
    }

    async fn drain(mut self, tcx: TyCtxt<'_>) -> Result<ObjectBuilder, ObjectError> {
        loop {
            match self {
                Self::Faulted { first } => return Err(first),
                Self::Streaming { window, builder } if window.is_empty() => return Ok(*builder),
                streaming => self = streaming.await_oldest(tcx).await,
            }
        }
    }
}

fn refuse(tcx: TyCtxt<'_>, symbol: &str, error: MirError) -> ! {
    tcx.dcx().fatal(format!("proxima-nova: {symbol}: {error}"))
}

// the cranelift pattern (rustc_codegen_cranelift base.rs:46,49): symbol_name, then instance_mir with no clone
fn lower_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    listing: Listing,
    allocs: &mut Vec<AllocId>,
) -> Function {
    let symbol = tcx.symbol_name(instance).name.to_string();
    let instance = mir::body_instance(instance);
    if !mir::has_body(tcx, instance) {
        refuse(tcx, &symbol, MirError::NoBody);
    }
    let body = tcx.instance_mir(instance.def);
    if !listing.any() {
        return mir::lower_body(tcx, instance, body, symbol.clone(), allocs)
            .unwrap_or_else(|error| refuse(tcx, &symbol, error));
    }
    let (function, lines) = mir::lower_body_listed(tcx, instance, body, symbol.clone(), allocs)
        .unwrap_or_else(|error| refuse(tcx, &symbol, error));
    if listing.layout {
        lines.iter().for_each(|line| println!("{line}"));
    }
    if listing.ir {
        println!("nova_ir fn={symbol} {function:?}");
    }
    function
}

#[derive(Clone, Copy)]
struct Listing {
    layout: bool,
    ir: bool,
}

impl Listing {
    fn any(self) -> bool {
        self.layout || self.ir
    }
}

// R27: lower on the rustc thread, run the backend on pool workers through a bounded window, feed in order
fn stream_crate<'tcx>(
    tcx: TyCtxt<'tcx>,
    instances: &[Instance<'tcx>],
    statics: &[DefId],
) -> ObjectBuilder {
    let settings = StreamSettings::from_env().unwrap_or_else(|error| {
        tcx.dcx()
            .fatal(format!("proxima-nova: stream settings: {error}"))
    });
    let pool =
        ProximaBackgroundPool::with_threads(settings.resolved_workers()).unwrap_or_else(|error| {
            tcx.dcx()
                .fatal(format!("proxima-nova: backend pool: {error}"))
        });
    let capacity = settings.window();
    let listing = Listing {
        layout: layout_mode(tcx.sess),
        ir: ir_mode(tcx.sess),
    };
    let minos = packed_deployment_target(tcx);
    let timed = time_mode(tcx.sess);
    let mut allocs = Vec::new();
    let streamed = block_on(async {
        let mut drive = Drive::new(minos);
        let mut lowering = Duration::ZERO;
        let mut lowered = 0usize;
        for instance in instances {
            let started = timed.then(Instant::now);
            let function = lower_instance(tcx, *instance, listing, &mut allocs);
            if let Some(started) = started {
                lowering += started.elapsed();
                lowered += 1;
            }
            drive = drive
                .submit(tcx, capacity, || {
                    pool.spawn_async(async move { Ok(backend().call(function).await) })
                        .map(|joined| joined.map_err(|error| error.to_string()))
                })
                .await;
        }
        if timed {
            println!(
                "nova_time lower_mir_s={:.3} functions={lowered}",
                lowering.as_secs_f64()
            );
        }
        drive.drain(tcx).await
    });
    match streamed {
        Ok(mut builder) => {
            feed_data(tcx, &mut builder, statics, allocs);
            builder
        }
        Err(error) => tcx.dcx().fatal(format!("proxima-nova: {error}")),
    }
}

impl CodegenBackend for NovaBackend {
    fn name(&self) -> &'static str {
        "proxima-nova"
    }

    fn target_config(&self, sess: &EarlySession) -> TargetConfig {
        let required = sess.target.abi_required_features().required;
        // rustc ties neon to fp-armv8 on aarch64 (rustc_target abi_required_features), and fp-armv8
        // is the target modifier that `core` records, so it must be reported alongside neon
        let features = internal_target_features::<0>(
            sess,
            |_feature| Default::default(),
            |feature| {
                required.contains(&feature) || (feature == "fp-armv8" && required.contains(&"neon"))
            },
        );
        TargetConfig {
            internal_target_features: features,
            has_reliable_f16: false,
            has_reliable_f16_math: false,
            has_reliable_f16b: false,
            has_reliable_f128: false,
            has_reliable_f128_math: false,
        }
    }

    fn target_cpu(&self, sess: &Session) -> String {
        sess.target.cpu.to_string()
    }

    fn codegen_crate(&self, tcx: TyCtxt<'_>) -> Box<dyn Any> {
        if list_mode(tcx.sess) {
            print_mono_items(tcx);
            // rustc's trait fixes the opaque return type
            return Box::new(CompiledModules {
                modules: vec![],
                allocator_module: None,
            });
        }
        let (name, instances, statics) = mono_instances(tcx);
        let builder = stream_crate(tcx, &instances, &statics);
        let bytes = builder
            .finish()
            .unwrap_or_else(|error| tcx.dcx().fatal(format!("proxima-nova: {error}")));
        let object = tcx
            .output_filenames(())
            .temp_path_for_cgu(OutputType::Object, &name);
        if let Err(error) = fs::write(&object, bytes) {
            tcx.dcx().fatal(format!(
                "proxima-nova: writing {}: {error}",
                object.display()
            ));
        }
        let module = CompiledModule {
            name,
            kind: ModuleKind::Regular,
            object: Some(object),
            global_asm_object: None,
            dwarf_object: None,
            bytecode: None,
            assembly: None,
            llvm_ir: None,
        };
        let compiled = CompiledModules {
            modules: vec![module],
            allocator_module: None,
        };
        // rustc treats a CompiledModules payload as finished codegen and skips join_codegen
        // (rustc_interface queries.rs:56-58), so the step its own write path takes (back/write.rs:2209)
        // runs here: it copies the cgu object to the -o / --emit path, as LLVM does
        produce_final_output_artifacts(tcx.sess, &compiled, tcx.output_filenames(()));
        // rustc's trait fixes the opaque return type
        Box::new(compiled)
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        _incr_comp_session: Option<&IncrCompSession>,
        _outputs: &OutputFilenames,
        _crate_info: &CrateInfo,
    ) -> (CompiledModules, WorkProductMap) {
        match ongoing_codegen.downcast::<CompiledModules>() {
            Ok(modules) => (*modules, WorkProductMap::default()),
            Err(_) => sess
                .dcx()
                .fatal("proxima-nova: join_codegen received a foreign codegen payload"),
        }
    }

    fn link(
        &self,
        sess: &Session,
        compiled_modules: CompiledModules,
        crate_info: CrateInfo,
        metadata: EncodedMetadata,
        outputs: &OutputFilenames,
    ) {
        if list_mode(sess) {
            // list mode produces no artifact, so there is nothing to link
            return;
        }
        link_binary(
            sess,
            &ArArchiveBuilderBuilder,
            compiled_modules,
            crate_info,
            metadata,
            outputs,
            self.name(),
        );
    }
}

/// rustc loads this symbol from the dylib; the trait object is the loader's fixed contract
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    Box::new(NovaBackend)
}

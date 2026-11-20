//! Optional CUDA volume evaluator for pure noise branches of the live chunk router.
mod abi;
mod blended;
mod fusion;
pub use blended::BlendedResident;
pub mod grid;
mod selection;
mod smear_table;
mod worker;

use crossbeam::channel::{Sender, bounded};
use cudarc::{
    driver::{
        CudaContext, CudaSlice, DeviceRepr,
        sys::{CUdevice_attribute, CUresult},
    },
    nvrtc::Ptx,
};
use pumpkin_config::world::{WorldgenBackend, WorldgenConfig};
use pumpkin_data::noise_router::{LinearOperation, UnaryOperation};
use pumpkin_util::noise::volume::DensityVolume;
use std::cell::Cell;
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

use super::super::router::{
    chunk_noise_router::ChunkNoiseFunctionComponent as Component,
    density_function::{NoiseFunctionComponentRange, noise::Noise},
    proto_noise_router::{
        DependentProtoNoiseFunctionComponent as Dep, IndependentProtoNoiseFunctionComponent as Leaf,
    },
};
use abi::CudaLayer;

pub type ResidentLayers = OnceLock<Option<Arc<CudaSlice<CudaLayer>>>>;
const DEFAULT_LANES: usize = 2;
const MAX_AUTO_LANES: usize = 4;
const MAX_LANES: usize = 8;
// Typical 3-chunk Overworld grid launch (~3 * 25 columns * ~10 planes).
const TYPICAL_BATCH_BLOCKS: usize = 768;
const MAX_BATCH: usize = 64;
const MAX_SAMPLES: usize = 1 << 20;
const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

static ACTIVE_DEVICE: AtomicUsize = AtomicUsize::new(0);

#[derive(Copy, Clone, Default)]
#[repr(C)]
struct Operation {
    kind: i32,
    a: f32,
    b: f32,
}

impl PartialEq for Operation {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.a.to_bits() == other.a.to_bits()
            && self.b.to_bits() == other.b.to_bits()
    }
}

#[derive(Copy, Clone, Default, PartialEq)]
#[repr(C)]
struct Tail {
    ops: [Operation; 8],
    count: i32,
}

const _: () = assert!(std::mem::size_of::<Tail>() == 100);
// SAFETY: Plain initialized scalar fields; layout is checked against worldgen.cu.
unsafe impl DeviceRepr for Tail {}

fn branch<'a>(stack: &'a [Component<'_>]) -> Option<(&'a Noise, Tail)> {
    let (leaf, tail) = sampler_branch(stack, false)?;
    let Leaf::Noise(noise) = leaf else {
        return None;
    };
    Some((noise, tail))
}

fn sampler_branch<'a>(mut stack: &'a [Component<'_>], terrain: bool) -> Option<(&'a Leaf, Tail)> {
    let mut tail = Tail::default();
    loop {
        let (node, prefix) = stack.split_last()?;
        let (child, kind, a, b) = match node {
            Component::Independent(leaf)
                if matches!(leaf, Leaf::Noise(_))
                    || (terrain && matches!(leaf, Leaf::InterpolatedNoise(_))) =>
            {
                return Some((leaf, tail));
            }
            Component::PassThrough(pass) => {
                stack = prefix.get(..=pass.input_index())?;
                continue;
            }
            Component::Dependent(Dep::Linear(node)) => {
                let kind = match node.data.operation {
                    LinearOperation::Add => 1,
                    LinearOperation::Mul => 2,
                };
                (node.input_index, kind, node.data.argument, 0.0)
            }
            Component::Dependent(Dep::Clamp(node)) => (
                node.input_index,
                7,
                node.data.min_value,
                node.data.max_value,
            ),
            Component::Dependent(Dep::Unary(node)) => {
                let (kind, factor) = match node.data.operation {
                    UnaryOperation::Abs => (3, 0.0),
                    UnaryOperation::Square => (4, 0.0),
                    UnaryOperation::Cube => (5, 0.0),
                    UnaryOperation::HalfNegative => (6, 0.5),
                    UnaryOperation::QuarterNegative => (6, 0.25),
                    UnaryOperation::Negate => (8, 0.0),
                    UnaryOperation::Squeeze if terrain => (9, 0.0),
                    UnaryOperation::Invert if terrain => (10, 0.0),
                    UnaryOperation::Sqrt if terrain && prefix[node.input_index].min() >= 0.0 => {
                        (11, 0.0)
                    }
                    UnaryOperation::Sign if terrain => (12, 0.0),
                    _ => return None,
                };
                (node.input_index, kind, factor, 0.0)
            }
            _ => return None,
        };
        *tail.ops.get_mut(tail.count as usize)? = Operation { kind, a, b };
        tail.count += 1;
        stack = prefix.get(..=child)?;
    }
}

struct Job {
    volume: DensityVolume,
    layers: Arc<CudaSlice<CudaLayer>>,
    tail: Tail,
    reply: Sender<Vec<f32>>,
}

enum Request {
    Volume(Job),
    Grid(grid::Job),
}

struct Engine {
    context: Arc<CudaContext>,
    jobs: Sender<Request>,
    disabled: Arc<AtomicBool>,
    verify: bool,
    grid: bool,
    pipeline: bool,
    terrain: bool,
    fuse_density: bool,
    select_density: bool,
    reserve_binary: bool,
    prefetch_early: bool,
    cache_early_plan: bool,
}

static ENGINE: OnceLock<Option<Engine>> = OnceLock::new();
static FIRST_RESULT: AtomicBool = AtomicBool::new(true);

static BATCHES: AtomicU64 = AtomicU64::new(0);
static VOLUMES: AtomicU64 = AtomicU64::new(0);
static FUSED: AtomicU64 = AtomicU64::new(0);
static BUSY: AtomicU64 = AtomicU64::new(0);
static TABLE_BYTES: AtomicU64 = AtomicU64::new(0);
static READBACK_BYTES: AtomicU64 = AtomicU64::new(0);
static GRID_BATCHES: AtomicU64 = AtomicU64::new(0);
static GRID_HITS: AtomicU64 = AtomicU64::new(0);
static GRID_CHUNKS: AtomicU64 = AtomicU64::new(0);
static PIPELINE_READY: AtomicU64 = AtomicU64::new(0);
static PIPELINE_WAITS: AtomicU64 = AtomicU64::new(0);
static PIPELINE_WAIT_NS: AtomicU64 = AtomicU64::new(0);
static BLENDED_HITS: AtomicU64 = AtomicU64::new(0);
static PAIRED_BLENDED_BATCHES: AtomicU64 = AtomicU64::new(0);
static SMEAR_TABLE_BATCHES: AtomicU64 = AtomicU64::new(0);
static SMEAR_TABLE_BUILDS: AtomicU64 = AtomicU64::new(0);
static FUSED_BINARY_HITS: AtomicU64 = AtomicU64::new(0);
static SELECTED_DENSITY_HITS: AtomicU64 = AtomicU64::new(0);
static EARLY_GRID_HITS: AtomicU64 = AtomicU64::new(0);
static CACHED_EARLY_GRID_HITS: AtomicU64 = AtomicU64::new(0);
static CACHED_PLAN_BUILDS: AtomicU64 = AtomicU64::new(0);
static SCHEDULED_GRID_SUBMISSIONS: AtomicU64 = AtomicU64::new(0);
static SCHEDULED_GRID_HITS: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub struct CudaWorldgenStats {
    pub available: bool,
    pub batches: u64,
    pub volumes: u64,
    pub fused_volumes: u64,
    pub queue_full_fallbacks: u64,
    pub sampler_upload_bytes: u64,
    pub result_readback_bytes: u64,
    pub grid_batches: u64,
    pub prefetched_hits: u64,
    pub grid_chunks: u64,
    pub pipeline_ready_at_first_use: u64,
    pub pipeline_waits: u64,
    pub pipeline_wait_ns: u64,
    pub blended_hits: u64,
    pub paired_blended_batches: u64,
    pub smear_table_batches: u64,
    pub smear_table_builds: u64,
    pub fused_binary_hits: u64,
    pub selected_density_hits: u64,
    pub early_grid_hits: u64,
    pub cached_early_grid_hits: u64,
    pub cached_plan_builds: u64,
    pub scheduled_grid_submissions: u64,
    pub scheduled_grid_hits: u64,
}

/// Lifetime counters for the live backend. Reading them does not initialize CUDA.
pub fn cuda_worldgen_stats() -> CudaWorldgenStats {
    CudaWorldgenStats {
        available: ENGINE
            .get()
            .and_then(Option::as_ref)
            .is_some_and(|e| !e.disabled.load(Ordering::Acquire)),
        batches: BATCHES.load(Ordering::Relaxed),
        volumes: VOLUMES.load(Ordering::Relaxed),
        fused_volumes: FUSED.load(Ordering::Relaxed),
        queue_full_fallbacks: BUSY.load(Ordering::Relaxed),
        sampler_upload_bytes: TABLE_BYTES.load(Ordering::Relaxed),
        result_readback_bytes: READBACK_BYTES.load(Ordering::Relaxed),
        grid_batches: GRID_BATCHES.load(Ordering::Relaxed),
        prefetched_hits: GRID_HITS.load(Ordering::Relaxed),
        grid_chunks: GRID_CHUNKS.load(Ordering::Relaxed),
        pipeline_ready_at_first_use: PIPELINE_READY.load(Ordering::Relaxed),
        pipeline_waits: PIPELINE_WAITS.load(Ordering::Relaxed),
        pipeline_wait_ns: PIPELINE_WAIT_NS.load(Ordering::Relaxed),
        blended_hits: BLENDED_HITS.load(Ordering::Relaxed),
        paired_blended_batches: PAIRED_BLENDED_BATCHES.load(Ordering::Relaxed),
        smear_table_batches: SMEAR_TABLE_BATCHES.load(Ordering::Relaxed),
        smear_table_builds: SMEAR_TABLE_BUILDS.load(Ordering::Relaxed),
        fused_binary_hits: FUSED_BINARY_HITS.load(Ordering::Relaxed),
        selected_density_hits: SELECTED_DENSITY_HITS.load(Ordering::Relaxed),
        early_grid_hits: EARLY_GRID_HITS.load(Ordering::Relaxed),
        cached_early_grid_hits: CACHED_EARLY_GRID_HITS.load(Ordering::Relaxed),
        cached_plan_builds: CACHED_PLAN_BUILDS.load(Ordering::Relaxed),
        scheduled_grid_submissions: SCHEDULED_GRID_SUBMISSIONS.load(Ordering::Relaxed),
        scheduled_grid_hits: SCHEDULED_GRID_HITS.load(Ordering::Relaxed),
    }
}

thread_local! {
    static CHECKING_CPU: Cell<bool> = const { Cell::new(false) };
}

struct CpuReferenceGuard;
impl Drop for CpuReferenceGuard {
    fn drop(&mut self) {
        CHECKING_CPU.set(false);
    }
}

#[derive(Clone, Copy)]
struct DeviceCapabilities {
    sm_count: Option<usize>,
    max_blocks_per_sm: Option<usize>,
    async_engines: Option<usize>,
    concurrent_kernels: Option<bool>,
    host_cpus: usize,
}

impl DeviceCapabilities {
    fn query(context: &CudaContext) -> Self {
        let attr = |key| {
            context
                .attribute(key)
                .ok()
                .and_then(|value| usize::try_from(value).ok())
        };
        Self {
            sm_count: attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                .filter(|&n| n > 0),
            max_blocks_per_sm: attr(
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
            )
            .filter(|&n| n > 0),
            async_engines: attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT),
            concurrent_kernels: attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS)
                .map(|v| v != 0),
            host_cpus: std::thread::available_parallelism()
                .map_or(DEFAULT_LANES * 2, std::num::NonZeroUsize::get),
        }
    }

    fn auto_streams(self) -> usize {
        let async_engines = self.async_engines.unwrap_or(1);
        let concurrent_kernels = self.concurrent_kernels.unwrap_or(true);
        if async_engines == 0 && !concurrent_kernels {
            return 1;
        }
        // One lane overlaps D2H readback and host dispatch while compute lanes run kernels.
        let desired = if concurrent_kernels && async_engines >= 2 {
            let compute_lanes = match (self.sm_count, self.max_blocks_per_sm) {
                (Some(sms), Some(blocks_per_sm)) => (sms.saturating_mul(blocks_per_sm)
                    / TYPICAL_BATCH_BLOCKS)
                    .clamp(1, MAX_AUTO_LANES - 1),
                _ => 1,
            };
            (compute_lanes + 1).min(MAX_AUTO_LANES)
        } else {
            DEFAULT_LANES
        };
        // Keep worker threads from starving the Rayon generation pool on small hosts.
        let host_cap = (self.host_cpus / 2).max(1);
        desired.min(host_cap)
    }
}

fn resolve_streams(caps: DeviceCapabilities, configured: usize) -> (usize, &'static str) {
    if configured > 0 {
        return (configured.min(MAX_LANES), "config");
    }
    (caps.auto_streams(), "auto")
}

fn start(device: usize, configured_streams: usize) -> Result<Engine, Box<dyn std::error::Error>> {
    // SAFETY: Probe the system driver through the same loader used by cudarc.
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return Err("NVIDIA CUDA driver library not found".into());
    }
    let count = usize::try_from(CudaContext::device_count()?).unwrap_or(0);
    if count == 0 {
        return Err("no CUDA-capable GPU detected".into());
    }
    if device >= count {
        return Err(if count == 1 {
            format!("cuda:{device} not found (only device 0 is available)").into()
        } else {
            format!("cuda:{device} not found (available devices: 0..{count})").into()
        });
    }

    let context = CudaContext::new(device)?;
    let gpu = context
        .name()
        .unwrap_or_else(|_| String::from("unknown GPU"));
    let arch = context.compute_capability().map_or_else(
        |_| String::from("unknown arch"),
        |(major, minor)| format!("sm_{major}{minor}"),
    );

    let module = context
        .load_module(Ptx::from_binary(
            include_bytes!(concat!(env!("OUT_DIR"), "/worldgen.fatbin")).to_vec(),
        ))
        .map_err(|error| {
            if matches!(
                error.0,
                CUresult::CUDA_ERROR_NO_BINARY_FOR_GPU | CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION
            ) {
                format!(
                    "cuda:{device} ({gpu}, {arch}) is not supported by this build's CUDA kernels ({error})"
                )
            } else {
                format!("failed to load CUDA kernels on cuda:{device} ({gpu}, {arch}): {error}")
            }
        })?;
    let kernel = module.load_function("fill_terrain_select")?;
    let prepare_table = module.load_function("prepare_terrain_y")?;
    let caps = DeviceCapabilities::query(&context);
    let (lanes, streams_source) = resolve_streams(caps, configured_streams);
    let (jobs, rx) = bounded(MAX_BATCH);
    let disabled = Arc::new(AtomicBool::new(false));
    for index in 0..lanes {
        let mut worker = worker::Lane::new(context.new_stream()?, kernel.clone(), false, true);
        worker.enable_selection();
        worker.enable_smear_table(prepare_table.clone());
        let receiver = rx.clone();
        let disabled = disabled.clone();
        std::thread::Builder::new()
            .name(format!("worldgen-cuda-{index}"))
            .spawn(move || {
                worker.run_grids(&receiver, &disabled, true, true);
            })?;
    }
    ACTIVE_DEVICE.store(device, Ordering::Relaxed);
    let sms = caps
        .sm_count
        .map(|n| format!("{n} SMs, "))
        .unwrap_or_default();
    let vram = context
        .mem_get_info()
        .map(|(free, total)| {
            format!(
                "{:.1} GiB VRAM, {:.1} GiB free, ",
                total as f64 / BYTES_PER_GIB,
                free as f64 / BYTES_PER_GIB,
            )
        })
        .unwrap_or_default();
    tracing::info!(
        "World generation uses CUDA on cuda:{device} ({gpu}, {arch}, {sms}{vram}{lanes} streams [{streams_source}])"
    );
    Ok(Engine {
        context,
        jobs,
        disabled,
        verify: false,
        grid: true,
        pipeline: true,
        terrain: true,
        fuse_density: true,
        select_density: true,
        reserve_binary: false,
        prefetch_early: true,
        cache_early_plan: true,
    })
}

fn disable(flag: &AtomicBool, reason: impl std::fmt::Display) {
    if !flag.swap(true, Ordering::AcqRel) {
        let device = ACTIVE_DEVICE.load(Ordering::Relaxed);
        tracing::warn!(
            "CUDA world generation disabled at runtime on cuda:{device}, falling back to CPU ({reason})"
        );
    }
}

fn start_from_config(config: WorldgenConfig) -> Result<Option<Engine>, Box<dyn std::error::Error>> {
    match config.backend {
        WorldgenBackend::Cuda => start(config.device, config.streams).map(Some),
        WorldgenBackend::Cpu | WorldgenBackend::Vulkan | WorldgenBackend::Metal => Ok(None),
    }
}

fn initialized_engine(config: WorldgenConfig) -> Option<&'static Engine> {
    ENGINE
        .get_or_init(|| match start_from_config(config) {
            Ok(Some(engine)) => Some(engine),
            Ok(None) => {
                tracing::info!("World generation uses CPU");
                None
            }
            Err(error) => {
                tracing::warn!("World generation falling back to CPU: {error}");
                None
            }
        })
        .as_ref()
}

pub(in crate::generation) fn initialize(config: WorldgenConfig) {
    let _ = initialized_engine(config);
}

pub fn scheduled_lookahead_enabled() -> bool {
    engine().is_some()
}

fn engine() -> Option<&'static Engine> {
    let engine = ENGINE.get()?.as_ref()?;
    (!engine.disabled.load(Ordering::Acquire)).then_some(engine)
}

fn resident_layers(engine: &Engine, noise: &Noise) -> Option<Arc<CudaSlice<CudaLayer>>> {
    let resident = noise.resident.cuda.get_or_init(|| {
        let layers: Vec<_> = noise
            .sampler
            .layers()
            .iter()
            .map(|l| CudaLayer {
                offset_x: l.noise.offset_x,
                offset_y: l.noise.offset_y,
                offset_z: l.noise.offset_z,
                xz_scale: noise.data.xz_scale * l.frequency,
                y_scale: noise.data.y_scale * l.frequency,
                amplitude: l.amplitude,
                permutations: *l.noise.permutations(),
                padding: [0; 4],
            })
            .collect();
        if layers.is_empty() || layers.len() * 28 > 48 * 1024 {
            return None;
        }
        let stream = engine.context.default_stream();
        match stream.clone_htod(&layers).and_then(|d| {
            stream.synchronize()?;
            Ok(d)
        }) {
            Ok(d) => {
                let bytes = std::mem::size_of_val(layers.as_slice());
                TABLE_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
                Some(Arc::new(d))
            }
            Err(e) => {
                disable(&engine.disabled, e);
                None
            }
        }
    });
    resident.clone()
}

fn submit(engine: &Engine, job: Request) -> bool {
    match engine.jobs.try_send(job) {
        Ok(()) => true,
        Err(crossbeam::channel::TrySendError::Full(_)) => {
            BUSY.fetch_add(1, Ordering::Relaxed);
            false
        }
        Err(crossbeam::channel::TrySendError::Disconnected(_)) => {
            disable(&engine.disabled, "worker queue disconnected");
            false
        }
    }
}

pub fn try_volume(stack: &mut [Component<'_>], out: &mut [f32], v: &DensityVolume) -> bool {
    // Tiny fallback volumes occur during scalar interpolation and do not amortize a launch.
    if CHECKING_CPU.get()
        || !(256..=MAX_SAMPLES).contains(&out.len())
        || out.len() != v.size()
        || v.size_z > 65535
    {
        return false;
    }
    let Some(engine) = engine() else { return false };
    if engine.grid {
        return grid::try_prefetched(engine, stack, out, v);
    }
    let Some((noise, tail)) = branch(stack) else {
        return false;
    };
    let Some(layers) = resident_layers(engine, noise) else {
        return false;
    };
    let id = noise.data.noise_id.id;
    let (reply, rx) = bounded(1);
    let job = Request::Volume(Job {
        volume: *v,
        layers,
        tail,
        reply,
    });
    if !submit(engine, job) {
        return false;
    }
    let Ok(values) = rx.recv() else { return false };
    apply_result(engine, Some(id), stack, out, v, &values)
}

fn apply_result(
    engine: &Engine,
    id: Option<usize>,
    stack: &mut [Component<'_>],
    out: &mut [f32],
    v: &DensityVolume,
    values: &[f32],
) -> bool {
    if engine.disabled.load(Ordering::Acquire) {
        return false;
    }
    if engine.verify {
        CHECKING_CPU.set(true);
        let _guard = CpuReferenceGuard;
        Component::sample_volume_from_stack(stack, out, v);
        if let Some(i) = values
            .iter()
            .zip(out.iter())
            .position(|(gpu, cpu)| gpu.to_bits() != cpu.to_bits())
        {
            disable(
                &engine.disabled,
                format_args!(
                    "noise {id:?} at {v:?} index {i}: GPU {:08x}, CPU {:08x}",
                    values[i].to_bits(),
                    out[i].to_bits()
                ),
            );
            return true; // out already holds the CPU result.
        }
    }
    out.copy_from_slice(values);
    if FIRST_RESULT.swap(false, Ordering::Relaxed) {
        tracing::debug!(
            samples = out.len(),
            verify = engine.verify,
            "First CUDA worldgen volume completed"
        );
    }
    true
}

#[cfg(test)]
mod tests;

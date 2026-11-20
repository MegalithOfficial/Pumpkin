//! Optional Vulkan compute volume evaluator for pure noise branches of the live chunk router.

mod abi;
mod device;
pub mod grid;

use abi::MAX_GRID_CHUNKS;
use crossbeam::channel::{Receiver, Sender, bounded};
use device::{ComputeLane, VulkanDevice};
use pumpkin_config::world::{WorldgenBackend, WorldgenConfig};
use pumpkin_util::noise::volume::DensityVolume;
use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

use super::super::router::chunk_noise_router::ChunkNoiseFunctionComponent as Component;

const MAX_BATCH: usize = 64;
const MAX_SAMPLES: usize = 1 << 20;

static ACTIVE_DEVICE: AtomicUsize = AtomicUsize::new(0);
static DISPATCHES: AtomicU64 = AtomicU64::new(0);
static HITS: AtomicU64 = AtomicU64::new(0);

struct Engine {
    device: Arc<VulkanDevice>,
    jobs: Sender<grid::Job>,
    disabled: Arc<AtomicBool>,
}

static ENGINE: OnceLock<Option<Engine>> = OnceLock::new();

thread_local! {
    static CHECKING_CPU: Cell<bool> = const { Cell::new(false) };
}

fn start(device_index: usize, configured_streams: usize) -> Result<Engine, String> {
    let (device, lanes, streams_source) = VulkanDevice::create(device_index, configured_streams)?;
    let (jobs, rx) = bounded(MAX_BATCH);
    let disabled = Arc::new(AtomicBool::new(false));
    for index in 0..lanes {
        let lane = ComputeLane::new(Arc::clone(&device), index)?;
        let receiver = rx.clone();
        let disabled = Arc::clone(&disabled);
        std::thread::Builder::new()
            .name(format!("worldgen-vulkan-{index}"))
            .spawn(move || run_worker(lane, &receiver, &disabled))
            .map_err(|e| format!("failed to spawn Vulkan worker thread: {e}"))?;
    }
    ACTIVE_DEVICE.store(device_index, Ordering::Relaxed);
    let vram = device.format_vram();
    tracing::info!(
        "World generation uses Vulkan on vulkan:{device_index} ({name}, Vulkan {api}, {vram}{lanes} streams [{streams_source}])",
        name = device.info.name,
        api = device.info.api_version,
    );
    Ok(Engine {
        device,
        jobs,
        disabled,
    })
}

fn run_worker(mut lane: ComputeLane, rx: &Receiver<grid::Job>, disabled: &AtomicBool) {
    let mut pending = VecDeque::with_capacity(MAX_GRID_CHUNKS);
    let mut batch = Vec::with_capacity(MAX_GRID_CHUNKS);
    loop {
        let first = match pending.pop_front() {
            Some(job) => job,
            None => match rx.recv() {
                Ok(job) => job,
                Err(_) => break,
            },
        };
        if disabled.load(Ordering::Acquire) {
            pending.clear();
            continue;
        }
        if first.is_cancelled() {
            continue;
        }
        pending.extend(rx.try_iter().take(MAX_GRID_CHUNKS - 1 - pending.len()));
        pending.retain(|job| !job.is_cancelled());

        let limit =
            (MAX_SAMPLES / (first.plane_count() * first.volume.size())).min(MAX_GRID_CHUNKS);
        while batch.len() + 1 < limit {
            let Some(index) = pending.iter().position(|job| compatible_grid(&first, job)) else {
                break;
            };
            if let Some(job) = pending.remove(index) {
                batch.push(job);
            }
        }
        batch.insert(0, first);
        if let Err(error) = dispatch_grid(&mut lane, &batch) {
            disable(disabled, error);
        }
        batch.clear();
    }
}

fn dispatch_grid(lane: &mut ComputeLane, batch: &[grid::Job]) -> Result<(), String> {
    let first = &batch[0];
    let v = first.volume;
    let samples_per_chunk = v.size() * first.plane_count();
    let total_len = samples_per_chunk * batch.len();
    let total_planes = first.plane_count() * batch.len();

    let mut params = first.plan.base_params;
    for (index, item) in batch.iter().enumerate() {
        params.origins[3 * index] = item.volume.min_block_x;
        params.origins[3 * index + 1] = item.volume.min_block_y;
        params.origins[3 * index + 2] = item.volume.min_block_z;
    }

    let raw = lane.execute(
        &first.plan.layers,
        &first.plan.smears,
        &params,
        v.size_x as u32,
        v.size_z as u32,
        total_planes as u32,
        total_len,
    )?;
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
    let values: Arc<[f32]> = Arc::from(raw);
    for (index, item) in batch.iter().enumerate() {
        let _ = item.reply.send(grid::Readback {
            values: Arc::clone(&values),
            offset: index * samples_per_chunk,
        });
    }
    Ok(())
}

fn compatible_grid(a: &grid::Job, b: &grid::Job) -> bool {
    a.volume.size_x == b.volume.size_x
        && a.volume.size_y == b.volume.size_y
        && a.volume.size_z == b.volume.size_z
        && a.volume.step_block_x == b.volume.step_block_x
        && a.volume.step_block_y == b.volume.step_block_y
        && a.volume.step_block_z == b.volume.step_block_z
        && Arc::ptr_eq(&a.plan, &b.plan)
}

fn disable(flag: &AtomicBool, reason: impl std::fmt::Display) {
    if !flag.swap(true, Ordering::AcqRel) {
        let device = ACTIVE_DEVICE.load(Ordering::Relaxed);
        tracing::warn!(
            "Vulkan world generation disabled at runtime on vulkan:{device}, falling back to CPU ({reason})"
        );
    }
}

fn initialized_engine(config: WorldgenConfig) -> Option<&'static Engine> {
    ENGINE
        .get_or_init(|| match config.backend {
            WorldgenBackend::Vulkan => match start(config.device, config.streams) {
                Ok(engine) => Some(engine),
                Err(error) => {
                    tracing::warn!("World generation falling back to CPU: {error}");
                    None
                }
            },
            WorldgenBackend::Cpu | WorldgenBackend::Cuda | WorldgenBackend::Metal => None,
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

fn submit(engine: &Engine, job: grid::Job) -> bool {
    match engine.jobs.try_send(job) {
        Ok(()) => true,
        Err(crossbeam::channel::TrySendError::Full(_)) => false,
        Err(crossbeam::channel::TrySendError::Disconnected(_)) => {
            disable(&engine.disabled, "worker queue disconnected");
            false
        }
    }
}

pub fn try_volume(stack: &[Component<'_>], out: &mut [f32], v: &DensityVolume) -> bool {
    if CHECKING_CPU.get()
        || !(256..=MAX_SAMPLES).contains(&out.len())
        || out.len() != v.size()
        || v.size_z > 65535
    {
        return false;
    }
    if engine().is_none() {
        return false;
    }
    grid::try_prefetched(stack, out, v)
}

#[cfg(test)]
mod tests;

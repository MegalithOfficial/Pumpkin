//! Optional Apple Metal compute volume evaluator for pure noise branches of the live chunk router.

#[allow(dead_code)]
mod abi;
#[cfg(target_os = "macos")]
mod device;
#[cfg(target_os = "macos")]
pub mod grid;

use pumpkin_config::world::WorldgenConfig;

#[cfg(not(target_os = "macos"))]
pub(in crate::generation) fn initialize(_config: WorldgenConfig) {
    tracing::warn!(
        "World generation falling back to CPU: Metal world generation is only supported on macOS"
    );
}

#[cfg(target_os = "macos")]
mod runtime {
    use super::abi::{
        CoordX, CoordY, CoordZ, MAX_GRID_CHUNKS, prepare_coord_x, prepare_coord_y, prepare_coord_z,
    };
    use super::device::{ComputeLane, MetalDevice};
    use super::grid;
    use crate::generation::noise::router::chunk_noise_router::ChunkNoiseFunctionComponent as Component;
    use crossbeam::channel::{Receiver, Sender, bounded};
    use pumpkin_config::world::{WorldgenBackend, WorldgenConfig};
    use pumpkin_util::noise::volume::DensityVolume;
    use std::cell::Cell;
    use std::collections::VecDeque;
    use std::sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    };

    const MAX_BATCH: usize = 64;
    pub(super) const MAX_SAMPLES: usize = 1 << 20;

    static ACTIVE_DEVICE: AtomicUsize = AtomicUsize::new(0);
    pub(super) static DISPATCHES: AtomicU64 = AtomicU64::new(0);
    pub(super) static HITS: AtomicU64 = AtomicU64::new(0);

    pub(super) struct Engine {
        pub(super) device: Arc<MetalDevice>,
        jobs: Sender<grid::Job>,
        disabled: Arc<AtomicBool>,
    }

    static ENGINE: OnceLock<Option<Engine>> = OnceLock::new();

    thread_local! {
        pub(super) static CHECKING_CPU: Cell<bool> = const { Cell::new(false) };
    }

    fn start(device_index: usize, configured_streams: usize) -> Result<Engine, String> {
        let (device, lanes, streams_source) =
            MetalDevice::create(device_index, configured_streams)?;
        let (jobs, rx) = bounded(MAX_BATCH);
        let disabled = Arc::new(AtomicBool::new(false));
        for index in 0..lanes {
            let lane = ComputeLane::new(Arc::clone(&device))?;
            let receiver = rx.clone();
            let disabled = Arc::clone(&disabled);
            std::thread::Builder::new()
                .name(format!("worldgen-metal-{index}"))
                .spawn(move || run_worker(lane, &receiver, &disabled))
                .map_err(|e| format!("failed to spawn Metal worker thread: {e}"))?;
        }
        ACTIVE_DEVICE.store(device_index, Ordering::Relaxed);
        let mem = device.format_memory();
        tracing::info!(
            "World generation uses Metal on metal:{device_index} ({name}, {mem}{lanes} streams [{streams_source}])",
            name = device.info.name,
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
        let mut coords_x = Vec::<CoordX>::new();
        let mut coords_z = Vec::<CoordZ>::new();
        let mut coords_y = Vec::<CoordY>::new();
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
                let Some(index) = pending.iter().position(|job| compatible_grid(&first, job))
                else {
                    break;
                };
                if let Some(job) = pending.remove(index) {
                    batch.push(job);
                }
            }
            batch.insert(0, first);
            if let Err(error) = dispatch_grid(
                &mut lane,
                &batch,
                &mut coords_x,
                &mut coords_z,
                &mut coords_y,
            ) {
                disable(disabled, error);
            }
            batch.clear();
        }
    }

    fn dispatch_grid(
        lane: &mut ComputeLane,
        batch: &[grid::Job],
        coords_x: &mut Vec<CoordX>,
        coords_z: &mut Vec<CoordZ>,
        coords_y: &mut Vec<CoordY>,
    ) -> Result<(), String> {
        let first = &batch[0];
        let v = first.volume;
        let samples_per_chunk = v.size() * first.plane_count();
        let total_len = samples_per_chunk * batch.len();
        let total_planes = first.plane_count() * batch.len();
        let layers = &first.plan.host_layers;

        // Deduplicate unique base Y origins across chunks in this batch.
        let mut base_ys = Vec::with_capacity(batch.len());
        let mut params = first.plan.base_params;
        params.chunk_count = batch.len() as u32;
        for (index, item) in batch.iter().enumerate() {
            let base_y = item.volume.min_block_y;
            let row = base_ys
                .iter()
                .position(|&b| b == base_y)
                .unwrap_or_else(|| {
                    base_ys.push(base_y);
                    base_ys.len() - 1
                });
            params.y_rows[index] = row as u32;
        }
        params.y_row_count = base_ys.len() as u32;

        // Precompute tiny 1D coordinate tables in exact f64 on the host.
        coords_x.clear();
        coords_x.reserve(layers.len() * batch.len() * v.size_x);
        coords_z.clear();
        coords_z.reserve(layers.len() * batch.len() * v.size_z);
        coords_y.clear();
        coords_y.reserve(layers.len() * base_ys.len() * v.size_y);

        for layer in layers {
            for item in batch {
                for ix in 0..v.size_x {
                    coords_x.push(prepare_coord_x(
                        layer,
                        item.volume.min_block_x + ix as i32 * v.step_block_x,
                    ));
                }
                for iz in 0..v.size_z {
                    coords_z.push(prepare_coord_z(
                        layer,
                        item.volume.min_block_z + iz as i32 * v.step_block_z,
                    ));
                }
            }
            for &base_y in &base_ys {
                for iy in 0..v.size_y {
                    coords_y.push(prepare_coord_y(layer, base_y + iy as i32 * v.step_block_y));
                }
            }
        }

        let raw = lane.execute(
            &first.plan.layers,
            coords_x,
            coords_z,
            coords_y,
            &params,
            v.size_x,
            v.size_z,
            total_planes,
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
                "Metal world generation disabled at runtime on metal:{device}, falling back to CPU ({reason})"
            );
        }
    }

    fn initialized_engine(config: WorldgenConfig) -> Option<&'static Engine> {
        ENGINE
            .get_or_init(|| match config.backend {
                WorldgenBackend::Metal => match start(config.device, config.streams) {
                    Ok(engine) => Some(engine),
                    Err(error) => {
                        tracing::warn!("World generation falling back to CPU: {error}");
                        None
                    }
                },
                WorldgenBackend::Cpu | WorldgenBackend::Cuda | WorldgenBackend::Vulkan => None,
            })
            .as_ref()
    }

    pub(in crate::generation) fn initialize(config: WorldgenConfig) {
        let _ = initialized_engine(config);
    }

    pub fn scheduled_lookahead_enabled() -> bool {
        engine().is_some()
    }

    pub(super) fn engine() -> Option<&'static Engine> {
        let engine = ENGINE.get()?.as_ref()?;
        (!engine.disabled.load(Ordering::Acquire)).then_some(engine)
    }

    pub(super) fn submit(engine: &Engine, job: grid::Job) -> bool {
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
}

#[cfg(target_os = "macos")]
pub(in crate::generation) use runtime::initialize;
#[cfg(target_os = "macos")]
use runtime::{CHECKING_CPU, DISPATCHES, Engine, HITS, MAX_SAMPLES, engine, submit};
#[cfg(target_os = "macos")]
pub use runtime::{scheduled_lookahead_enabled, try_volume};

#[cfg(test)]
mod tests;

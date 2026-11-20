//! Precompute regular noise branches that share an interpolation grid.
use super::super::super::router::chunk_density_function::ChunkSpecificNoiseFunctionComponent;
use super::super::super::router::{
    chunk_noise_router::ChunkNoiseRouter, proto_noise_router::ProtoNoiseRouter,
};
use super::{
    BLENDED_HITS, CACHED_EARLY_GRID_HITS, CACHED_PLAN_BUILDS, CHECKING_CPU, Component, Dep,
    EARLY_GRID_HITS, Engine, FUSED_BINARY_HITS, GRID_HITS, Leaf, MAX_SAMPLES, PIPELINE_READY,
    PIPELINE_WAIT_NS, PIPELINE_WAITS, Request, SELECTED_DENSITY_HITS, Tail, apply_result, blended,
    engine, fusion, resident_layers, sampler_branch, selection, submit,
};
use super::{SCHEDULED_GRID_HITS, SCHEDULED_GRID_SUBMISSIONS};
use crate::generation::generator::VanillaGenerator;
use crossbeam::channel::{Receiver, Sender, TryRecvError, bounded};
use cudarc::driver::CudaSlice;
use pumpkin_util::noise::volume::DensityVolume;
use std::{
    cell::{Cell, OnceCell, RefCell},
    marker::PhantomData,
    rc::Rc,
    sync::Arc,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

const BASE_PLANES: usize = 24;
pub(super) const MAX_PLANES: usize = BASE_PLANES + 1;
const MAX_VISITS: usize = 512;

pub(super) struct Plane {
    pub layers: Arc<CudaSlice<super::abi::CudaLayer>>,
    pub tail: Tail,
    pub companion: Option<fusion::Operand>,
    pub selection: Option<selection::Selection>,
}

pub struct Plan {
    pub(super) volume: DensityVolume,
    pub(super) indices: Vec<(usize, Option<usize>)>,
    pub(super) fused_mask: u32,
    pub(super) selection_mask: u32,
    pub(super) planes: Vec<Plane>,
    pub(super) blended: Option<blended::Plane>,
}

pub(super) struct Job {
    pub volume: DensityVolume,
    pub plan: Arc<Plan>,
    pub reply: Sender<Readback>,
    pub cancel: Option<Arc<AtomicBool>>,
}

impl Job {
    pub(super) fn plane_count(&self) -> usize {
        self.plan.planes.len() + usize::from(self.plan.blended.is_some())
    }

    pub(super) fn is_cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(Ordering::Acquire))
    }
}

pub(super) struct Readback {
    pub values: Arc<[f32]>,
    pub offset: usize,
}

enum ResultState {
    Ready(Readback),
    Pending {
        receiver: Receiver<Readback>,
        result: OnceCell<Option<Readback>>,
    },
}

impl ResultState {
    fn get(&self) -> Option<&Readback> {
        match self {
            Self::Ready(readback) => Some(readback),
            Self::Pending { receiver, result } => result
                .get_or_init(|| match receiver.try_recv() {
                    Ok(readback) => {
                        PIPELINE_READY.fetch_add(1, Ordering::Relaxed);
                        Some(readback)
                    }
                    Err(TryRecvError::Empty) => {
                        let start = Instant::now();
                        let readback = receiver.recv().ok();
                        PIPELINE_WAITS.fetch_add(1, Ordering::Relaxed);
                        PIPELINE_WAIT_NS
                            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        readback
                    }
                    Err(TryRecvError::Disconnected) => None,
                })
                .as_ref(),
        }
    }
}

struct Prefetched {
    base: usize,
    volume: DensityVolume,
    plan: Arc<Plan>,
    result: ResultState,
    used: Cell<u32>,
    early: bool,
    cached: bool,
    scheduled: bool,
}

pub struct PreparedGrid {
    target: DensityVolume,
    batch: Prefetched,
}

impl PreparedGrid {
    pub fn activate(self, generator: &VanillaGenerator) -> Option<EarlyGridScope<'_>> {
        if CHECKING_CPU.get() || EARLY_GRID.with_borrow(Option::is_some) {
            return None;
        }
        EARLY_GRID.set(Some(EarlyGrid {
            source: &raw const generator.base_router.noise,
            target: self.target,
            batch: Some(self.batch),
        }));
        Some(EarlyGridScope(PhantomData))
    }
}

struct EarlyGrid {
    source: *const ProtoNoiseRouter,
    target: DensityVolume,
    batch: Option<Prefetched>,
}

impl EarlyGrid {
    fn bind(
        &mut self,
        source: &ProtoNoiseRouter,
        target: &DensityVolume,
        stack_base: usize,
    ) -> Option<Prefetched> {
        if !std::ptr::eq(self.source, source) || self.target != *target {
            return None;
        }
        let mut batch = self.batch.take()?;
        batch.base = stack_base;
        batch.early = true;
        Some(batch)
    }
}

thread_local! {
    static PREFETCHED: RefCell<Option<Prefetched>> = const { RefCell::new(None) };
    static EARLY_GRID: RefCell<Option<EarlyGrid>> = const { RefCell::new(None) };
}

pub struct EarlyGridScope<'a>(PhantomData<(&'a ProtoNoiseRouter, Rc<()>)>);

impl Drop for EarlyGridScope<'_> {
    fn drop(&mut self) {
        drop(EARLY_GRID.take());
    }
}

struct ClearPrefetched;
impl Drop for ClearPrefetched {
    fn drop(&mut self) {
        if let Some(batch) = PREFETCHED.take() {
            let used = batch.used.get();
            if used != 0 && batch.early {
                EARLY_GRID_HITS.fetch_add(1, Ordering::Relaxed);
                if batch.cached {
                    CACHED_EARLY_GRID_HITS.fetch_add(1, Ordering::Relaxed);
                }
                if batch.scheduled {
                    SCHEDULED_GRID_HITS.fetch_add(1, Ordering::Relaxed);
                }
            }
            GRID_HITS.fetch_add(u64::from(used.count_ones()), Ordering::Relaxed);
            let fused = (used & batch.plan.fused_mask).count_ones();
            if fused > 0 {
                FUSED_BINARY_HITS.fetch_add(u64::from(fused), Ordering::Relaxed);
            }
            let selected = (used & batch.plan.selection_mask).count_ones();
            if selected > 0 {
                SELECTED_DENSITY_HITS.fetch_add(u64::from(selected), Ordering::Relaxed);
            }
            if let Some((_, last)) = batch.plan.indices.last()
                && last.is_none()
                && used & (1u32 << (batch.plan.indices.len() - 1)) != 0
            {
                BLENDED_HITS.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

struct CandidateGrid {
    volume: DensityVolume,
    indices: Vec<usize>,
    blended: Option<usize>,
    fused: usize,
    regular_leaves: usize,
    selections: usize,
}

#[expect(clippy::too_many_lines, clippy::fn_params_excessive_bools)]
fn candidates(
    stack: &[Component<'_>],
    volume: &DensityVolume,
    terrain: bool,
    fuse_density: bool,
    select_density: bool,
    reserve_binary: bool,
) -> Option<CandidateGrid> {
    let mut visited = Vec::new();
    let mut pending = vec![(stack.len().checked_sub(1)?, *volume)];
    let mut grids: Vec<CandidateGrid> = Vec::new();
    for _ in 0..MAX_VISITS {
        let Some((index, volume)) = pending.pop() else {
            break;
        };
        let seen = if let Some((_, seen)) = visited.iter_mut().find(|(key, _)| *key == volume) {
            seen
        } else {
            visited.push((volume, vec![false; stack.len()]));
            &mut visited.last_mut()?.1
        };
        if std::mem::replace(&mut seen[index], true) {
            continue;
        }
        let volume_stack = &stack[..=index];
        let sampler = sampler_branch(volume_stack, terrain);
        let can_fuse = fuse_density && sampler.is_none() && fusion::branch(volume_stack).is_some();
        let selection = if select_density && sampler.is_none() && !can_fuse {
            selection::branch(volume_stack)
        } else {
            None
        };
        if sampler.is_some() || can_fuse || selection.is_some() {
            let size = volume.size();
            if !(256..=MAX_SAMPLES / 2).contains(&size) || volume.size_z > 65535 {
                continue;
            }
            let grid = if let Some(grid) = grids.iter_mut().find(|grid| grid.volume == volume) {
                grid
            } else {
                grids.push(CandidateGrid {
                    volume,
                    indices: Vec::new(),
                    blended: None,
                    fused: 0,
                    regular_leaves: 0,
                    selections: 0,
                });
                grids.last_mut()?
            };
            let limit = MAX_SAMPLES / size - usize::from(reserve_binary);
            if let Some(selection) = selection {
                let leaves = grid.regular_leaves + selection.sources();
                if grid.selections < selection::MAX_SELECTIONS
                    && leaves <= BASE_PLANES
                    && leaves + usize::from(grid.blended.is_some()) <= limit
                {
                    grid.indices.push(index);
                    grid.regular_leaves = leaves;
                    grid.selections += 1;
                    continue;
                }
            } else if can_fuse {
                // The first pair in hybrid mode uses the reserved output descriptor.
                // Its right sampler already lives in the separate binary descriptor array.
                let uses_reservation = reserve_binary && grid.fused == 0;
                let leaves = grid.regular_leaves + if uses_reservation { 0 } else { 2 };
                if grid.fused < fusion::MAX_PAIRS
                    && leaves <= BASE_PLANES
                    && leaves + usize::from(grid.blended.is_some()) <= limit
                {
                    grid.indices.push(index);
                    grid.regular_leaves = leaves;
                    grid.fused += 1;
                    continue;
                }
                // The pair exceeds the budget, so try its children as individual leaves.
            } else {
                if matches!(sampler, Some((Leaf::InterpolatedNoise(_), _))) {
                    if grid.regular_leaves < limit {
                        grid.blended.get_or_insert(index);
                    }
                } else if grid.regular_leaves < BASE_PLANES
                    && grid.regular_leaves + usize::from(grid.blended.is_some()) < limit
                {
                    grid.indices.push(index);
                    grid.regular_leaves += 1;
                }
                continue;
            }
        }
        let mut push = |child| pending.push((child, volume));
        match &stack[index] {
            Component::PassThrough(pass) => push(pass.input_index()),
            Component::Chunk(ChunkSpecificNoiseFunctionComponent::Cache(cache))
                if cache.needs_volume(&volume) =>
            {
                push(cache.input_index);
            }
            Component::Chunk(ChunkSpecificNoiseFunctionComponent::Interpolated(interpolated)) => {
                if let Some(corners) = interpolated.input_volume(&volume) {
                    pending.push((interpolated.input_index, corners));
                }
            }
            Component::Dependent(Dep::Binary(node)) => {
                push(node.input2_index);
                push(node.input1_index);
            }
            Component::Dependent(Dep::RangeChoice(node)) => {
                push(node.when_out_index);
                push(node.when_in_index);
                push(node.input_index);
            }
            Component::Dependent(Dep::IntervalSelect(node)) => {
                for &child in node.functions_indices.iter().rev() {
                    push(child);
                }
                push(node.input_index);
            }
            Component::Dependent(Dep::Lerp(node)) => {
                push(node.second_index);
                push(node.first_index);
                push(node.alpha_index);
            }
            Component::Dependent(Dep::Linear(node)) => push(node.input_index),
            Component::Dependent(Dep::Unary(node)) => push(node.input_index),
            Component::Dependent(Dep::Clamp(node)) => push(node.input_index),
            Component::Dependent(Dep::Rounding(node)) => {
                push(node.multiple_index);
                push(node.input_index);
            }
            Component::Dependent(Dep::ShiftedNoise(node)) => {
                push(node.input_z_index);
                push(node.input_y_index);
                push(node.input_x_index);
            }
            _ => {}
        }
    }
    let mut grid = grids
        .into_iter()
        .max_by_key(|grid| (grid.blended.is_some(), grid.regular_leaves))?;
    grid.indices.sort_unstable();
    Some(grid)
}

fn prepare_plan(engine: &Engine, stack: &[Component<'_>], volume: &DensityVolume) -> Option<Plan> {
    let CandidateGrid {
        volume,
        indices,
        blended,
        ..
    } = candidates(
        stack,
        volume,
        engine.terrain,
        engine.fuse_density,
        engine.select_density,
        engine.reserve_binary,
    )?;
    if indices.len() < 2 && blended.is_none() {
        return None;
    }
    let mut planes = Vec::with_capacity(indices.len());
    let mut keys = Vec::with_capacity(indices.len() + usize::from(blended.is_some()));
    let mut fused_mask = 0u32;
    let mut selection_mask = 0u32;
    for index in indices {
        let volume_stack = &stack[..=index];
        if engine.select_density
            && let Some(selection) = selection::branch(volume_stack)
        {
            let Some(plane) = selection.upload(engine) else {
                continue;
            };
            selection_mask |= 1u32 << planes.len();
            keys.push((index, Some(selection.selector.0.data.noise_id.id)));
            planes.push(plane);
        } else if engine.fuse_density
            && let Some(pair) = fusion::branch(volume_stack)
        {
            let Some(layers) = resident_layers(engine, pair.left.0) else {
                continue;
            };
            let Some(right) = resident_layers(engine, pair.right.0) else {
                continue;
            };
            if (layers.len() + right.len()) > super::worker::MAX_COLUMN_STATES {
                continue;
            }
            fused_mask |= 1u32 << planes.len();
            keys.push((index, Some(pair.left.0.data.noise_id.id)));
            planes.push(Plane {
                layers,
                tail: pair.left.1,
                companion: Some(fusion::Operand {
                    layers: right,
                    tail: pair.right.1,
                    operation: pair.operation,
                }),
                selection: None,
            });
        } else {
            let Some((Leaf::Noise(noise), tail)) = sampler_branch(volume_stack, engine.terrain)
            else {
                continue;
            };
            let Some(layers) = resident_layers(engine, noise) else {
                continue;
            };
            keys.push((index, Some(noise.data.noise_id.id)));
            planes.push(Plane {
                layers,
                tail,
                companion: None,
                selection: None,
            });
        }
    }
    let blended = blended.and_then(|index| {
        let (Leaf::InterpolatedNoise(sampler), tail) = sampler_branch(&stack[..=index], true)?
        else {
            return None;
        };
        let layers = blended::resident(engine, sampler)?;
        keys.push((index, None));
        Some(blended::Plane { layers, tail })
    });
    if (planes.len() < 2 && blended.is_none()) || engine.disabled.load(Ordering::Acquire) {
        return None;
    }
    Some(Plan {
        volume,
        indices: keys,
        fused_mask,
        selection_mask,
        planes,
        blended,
    })
}

fn enqueue(
    engine: &Engine,
    plan: Arc<Plan>,
    volume: DensityVolume,
    base: usize,
    cancel: Option<Arc<AtomicBool>>,
) -> Option<Prefetched> {
    if engine.disabled.load(Ordering::Acquire)
        || cancel
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Acquire))
    {
        return None;
    }
    let scheduled = cancel.is_some();
    let (reply, rx) = bounded(1);
    if !submit(
        engine,
        Request::Grid(Job {
            volume,
            plan: plan.clone(),
            reply,
            cancel,
        }),
    ) {
        return None;
    }
    if scheduled {
        SCHEDULED_GRID_SUBMISSIONS.fetch_add(1, Ordering::Relaxed);
    }
    let result = if engine.pipeline {
        ResultState::Pending {
            receiver: rx,
            result: OnceCell::new(),
        }
    } else {
        ResultState::Ready(rx.recv().ok()?)
    };
    Some(Prefetched {
        base,
        volume,
        plan,
        result,
        used: Cell::new(0),
        early: false,
        cached: false,
        scheduled,
    })
}

pub fn prefetch_chunk(
    generator: &VanillaGenerator,
    chunk_x: i32,
    chunk_z: i32,
) -> Option<EarlyGridScope<'_>> {
    if CHECKING_CPU.get() || EARLY_GRID.with_borrow(Option::is_some) {
        return None;
    }
    prepare_chunk(generator, chunk_x, chunk_z, None)?.activate(generator)
}

pub fn prepare_chunk(
    generator: &VanillaGenerator,
    chunk_x: i32,
    chunk_z: i32,
    cancel: Option<Arc<AtomicBool>>,
) -> Option<PreparedGrid> {
    let engine = engine().filter(|e| e.prefetch_early && !e.disabled.load(Ordering::Acquire))?;
    let source = &generator.base_router.noise;
    let target = generator.noise_density_volume(chunk_x, chunk_z);
    let cached_plan = if engine.cache_early_plan {
        generator
            .gpu_plan
            .cuda
            .get_or_init(|| prepare_cached_plan(engine, generator))
            .as_ref()
    } else {
        None
    };
    let batch = if let Some(plan) = cached_plan
        && let Some(volume) = translate_volume(plan.volume, &target)
    {
        let mut batch = enqueue(engine, plan.clone(), volume, 0, cancel)?;
        batch.cached = true;
        batch
    } else {
        let router = ChunkNoiseRouter::generate_for_prefetch(source);
        let plan = prepare_plan(engine, router.final_density_stack(), &target)?;
        let volume = plan.volume;
        enqueue(engine, Arc::new(plan), volume, 0, cancel)?
    };
    Some(PreparedGrid { target, batch })
}

fn prepare_cached_plan(engine: &Engine, generator: &VanillaGenerator) -> Option<Arc<Plan>> {
    // The planner only offloads pure branches. Structure contributions are still
    // evaluated by the chunk router after references are ready.
    let router = ChunkNoiseRouter::generate_for_prefetch(&generator.base_router.noise);
    let stack = router.final_density_stack();
    if !supports_chunk_translation(stack) {
        return None;
    }
    let plan = prepare_plan(engine, stack, &generator.noise_density_volume(0, 0))?;
    CACHED_PLAN_BUILDS.fetch_add(1, Ordering::Relaxed);
    Some(Arc::new(plan))
}

fn supports_chunk_translation(stack: &[Component<'_>]) -> bool {
    stack.iter().all(|node| match node {
        Component::Chunk(ChunkSpecificNoiseFunctionComponent::Interpolated(interpolated)) => {
            interpolated.supports_chunk_translation()
        }
        _ => true,
    })
}

fn translate_volume(mut volume: DensityVolume, target: &DensityVolume) -> Option<DensityVolume> {
    volume.min_block_x = volume.min_block_x.checked_add(target.min_block_x)?;
    volume.min_block_z = volume.min_block_z.checked_add(target.min_block_z)?;
    Some(volume)
}

pub fn try_sample_density(
    stack: &mut [Component<'_>],
    out: &mut [f32],
    volume: &DensityVolume,
    source: &ProtoNoiseRouter,
) -> bool {
    if CHECKING_CPU.get() || PREFETCHED.with_borrow(Option::is_some) {
        return false;
    }
    let Some(engine) = engine().filter(|e| e.grid) else {
        return false;
    };
    let early = if engine.prefetch_early {
        EARLY_GRID
            .with_borrow_mut(|slot| slot.as_mut()?.bind(source, volume, stack.as_ptr() as usize))
    } else {
        None
    };
    let Some(batch) = early.or_else(|| {
        let plan = prepare_plan(engine, stack, volume)?;
        let sampled = plan.volume;
        enqueue(
            engine,
            Arc::new(plan),
            sampled,
            stack.as_ptr() as usize,
            None,
        )
    }) else {
        return false;
    };
    PREFETCHED.set(Some(batch));
    let _prefetched = ClearPrefetched;
    Component::sample_volume_from_stack(stack, out, volume);
    true
}
pub(super) fn try_prefetched(
    engine: &Engine,
    stack: &mut [Component<'_>],
    out: &mut [f32],
    v: &DensityVolume,
) -> bool {
    PREFETCHED.with_borrow(|state| {
        let Some(batch) = state else { return false };
        if batch.volume != *v || batch.base != stack.as_ptr() as usize {
            return false;
        }
        let Some(index) = stack.len().checked_sub(1) else {
            return false;
        };
        let indices = &batch.plan.indices;
        let regular_count =
            indices.len() - usize::from(indices.last().is_some_and(|(_, id)| id.is_none()));
        let plane = if regular_count < indices.len() && indices[regular_count].0 == index {
            regular_count
        } else if let Ok(plane) =
            indices[..regular_count].binary_search_by_key(&index, |&(key, _)| key)
        {
            plane
        } else {
            return false;
        };
        let Some(result) = batch.result.get() else {
            return false;
        };
        let start = result.offset + plane * out.len();
        let used = apply_result(
            engine,
            indices[plane].1,
            stack,
            out,
            v,
            &result.values[start..start + out.len()],
        );
        if used {
            batch.used.set(batch.used.get() | (1u32 << plane));
        }
        used
    })
}

#[cfg(test)]
pub(super) fn candidate_count(stack: &[Component<'_>], volume: &DensityVolume) -> usize {
    candidates(stack, volume, false, false, false, false).map_or(0, |grid| grid.indices.len())
}

#[cfg(test)]
pub(super) fn blended_grid(
    stack: &[Component<'_>],
    volume: &DensityVolume,
) -> Option<DensityVolume> {
    let grid = candidates(stack, volume, true, false, false, false)?;
    grid.blended.map(|_| grid.volume)
}

#[cfg(test)]
pub(super) fn fused_candidate_count(stack: &[Component<'_>], volume: &DensityVolume) -> usize {
    candidates(stack, volume, true, true, false, false).map_or(0, |grid| grid.fused)
}

#[cfg(test)]
pub(super) fn selection_counts(
    stack: &[Component<'_>],
    volume: &DensityVolume,
    reserve_binary: bool,
) -> (usize, usize, usize) {
    candidates(stack, volume, true, true, true, reserve_binary).map_or((0, 0, 0), |grid| {
        (
            grid.selections,
            grid.fused,
            grid.indices.len() + usize::from(grid.blended.is_some()),
        )
    })
}

#[cfg(test)]
#[test]
fn early_batch_can_only_bind_once_to_its_router_and_volume() {
    use crate::generation::{
        GlobalRandomConfig, noise::router::proto_noise_router::ProtoNoiseRouters,
    };
    use pumpkin_data::noise_router::OVERWORLD_BASE_NOISE_ROUTER;

    let original = ProtoNoiseRouters::generate(
        &OVERWORLD_BASE_NOISE_ROUTER,
        &GlobalRandomConfig::new(42, false),
    );
    let other = ProtoNoiseRouters::generate(
        &OVERWORLD_BASE_NOISE_ROUTER,
        &GlobalRandomConfig::new(43, false),
    );
    let target = DensityVolume::with_block_step(16, 384, 16, -112, -64, 144);
    let neighbor = DensityVolume::with_block_step(16, 384, 16, -96, -64, 144);
    let corners = DensityVolume::new(5, 49, 5, -112, -64, 144, 4, 8, 4);
    let mut pending = EarlyGrid {
        source: &raw const original.noise,
        target,
        batch: Some(Prefetched {
            base: 0,
            volume: corners,
            plan: Arc::new(Plan {
                volume: corners,
                indices: Vec::new(),
                fused_mask: 0,
                selection_mask: 0,
                planes: Vec::new(),
                blended: None,
            }),
            result: ResultState::Ready(Readback {
                values: Arc::from([]),
                offset: 0,
            }),
            used: Cell::new(0),
            early: false,
            cached: false,
            scheduled: false,
        }),
    };
    assert!(pending.bind(&other.noise, &target, 1).is_none());
    assert!(pending.bind(&original.noise, &neighbor, 1).is_none());
    let batch = pending
        .bind(&original.noise, &target, 2)
        .expect("matching request");
    assert_eq!(batch.base, 2);
    assert_eq!(batch.volume, corners);
    assert!(batch.early);
    assert!(pending.bind(&original.noise, &target, 2).is_none());
}

#[cfg(test)]
#[test]
fn translated_plan_matches_fresh_plans_at_other_chunk_origins() {
    use crate::generation::{
        GlobalRandomConfig,
        noise::router::{
            chunk_density_function::Interpolated, proto_noise_router::ProtoNoiseRouters,
        },
    };
    use pumpkin_data::noise_router::OVERWORLD_BASE_NOISE_ROUTER;

    let base = ProtoNoiseRouters::generate(
        &OVERWORLD_BASE_NOISE_ROUTER,
        &GlobalRandomConfig::new(42, false),
    );
    let router = ChunkNoiseRouter::generate_for_prefetch(&base.noise);
    let stack = router.final_density_stack();
    assert!(supports_chunk_translation(stack));
    assert!(!supports_chunk_translation(&[Component::Chunk(
        ChunkSpecificNoiseFunctionComponent::Interpolated(Interpolated::new(0, 3, 8, -1.0, 1.0)),
    )]));

    let target = DensityVolume::with_block_step(16, 384, 16, 0, -64, 0);
    let template = candidates(stack, &target, true, true, true, false).expect("origin plan");
    assert!(template.blended.is_some());
    for (x, z) in [(-112, 144), (29_989_984, -29_989_984)] {
        let target = DensityVolume {
            min_block_x: x,
            min_block_z: z,
            ..target
        };
        let fresh = candidates(stack, &target, true, true, true, false).expect("chunk plan");
        assert_eq!(
            translate_volume(template.volume, &target),
            Some(fresh.volume)
        );
        assert_eq!(template.indices, fresh.indices);
        assert_eq!(template.blended, fresh.blended);
    }
}

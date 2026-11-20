//! Precompute regular noise branches that share an interpolation grid on Vulkan.

use super::super::super::router::{
    chunk_density_function::ChunkSpecificNoiseFunctionComponent,
    chunk_noise_router::{ChunkNoiseFunctionComponent as Component, ChunkNoiseRouter},
    density_function::{
        NoiseFunctionComponentRange,
        noise::{InterpolatedNoiseSampler, Noise},
    },
    proto_noise_router::{
        DependentProtoNoiseFunctionComponent as Dep,
        IndependentProtoNoiseFunctionComponent as Leaf, ProtoNoiseRouter,
    },
};
use super::{
    CHECKING_CPU, Engine, MAX_SAMPLES,
    abi::{
        BatchParams, BlendedDesc, MAX_ALTERNATIVES, MAX_BINARY_PAIRS, MAX_COLUMN_STATES,
        MAX_PLANES, MAX_SELECTIONS, Operation, PlaneDesc, SelectorDesc, Tail, VkLayer,
    },
    device::StorageBuffer,
    engine, submit,
};
use crate::generation::generator::VanillaGenerator;
use crossbeam::channel::{Receiver, Sender, bounded};
use pumpkin_data::noise_router::{BinaryOperation, LinearOperation, UnaryOperation};
use pumpkin_util::noise::volume::DensityVolume;
use std::{
    cell::{Cell, OnceCell, RefCell},
    marker::PhantomData,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub struct Plan {
    pub(super) volume: DensityVolume,
    pub(super) indices: Vec<(usize, Option<usize>)>,
    pub(super) plane_count: usize,
    pub(super) layers: StorageBuffer<VkLayer>,
    pub(super) smears: StorageBuffer<f64>,
    pub(super) base_params: BatchParams,
}

pub(super) struct Job {
    pub volume: DensityVolume,
    pub plan: Arc<Plan>,
    pub reply: Sender<Readback>,
    pub cancel: Option<Arc<AtomicBool>>,
}

impl Job {
    pub(super) fn plane_count(&self) -> usize {
        self.plan.plane_count
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

struct ResultState {
    rx: Receiver<Readback>,
    values: OnceCell<Option<Readback>>,
}

impl ResultState {
    const fn new(rx: Receiver<Readback>) -> Self {
        Self {
            rx,
            values: OnceCell::new(),
        }
    }

    fn get(&self) -> Option<&Readback> {
        self.values.get_or_init(|| self.rx.recv().ok()).as_ref()
    }
}

struct Prefetched {
    base: usize,
    volume: DensityVolume,
    plan: Arc<Plan>,
    result: ResultState,
    used: Cell<u32>,
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
        drop(PREFETCHED.take());
    }
}

fn sampler_branch<'a>(mut stack: &'a [Component<'_>]) -> Option<(&'a Leaf, Tail)> {
    let mut tail = Tail::default();
    loop {
        let (node, prefix) = stack.split_last()?;
        let (child, kind, a, b) = match node {
            Component::Independent(leaf)
                if matches!(leaf, Leaf::Noise(_) | Leaf::InterpolatedNoise(_)) =>
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
                    UnaryOperation::Squeeze => (9, 0.0),
                    UnaryOperation::Invert => (10, 0.0),
                    UnaryOperation::Sqrt if prefix[node.input_index].min() >= 0.0 => (11, 0.0),
                    UnaryOperation::Sign => (12, 0.0),
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

struct BinaryBranch<'a> {
    left: (&'a Noise, Tail),
    right: (&'a Noise, Tail),
    operation: u32,
}

fn binary_branch<'a>(stack: &'a [Component<'_>]) -> Option<BinaryBranch<'a>> {
    let (Component::Dependent(Dep::Binary(node)), prefix) = stack.split_last()? else {
        return None;
    };
    let lhs = prefix.get(..=node.input1_index)?;
    let rhs = prefix.get(..=node.input2_index)?;
    let operation = match node.data.operation {
        BinaryOperation::Add => 1,
        BinaryOperation::Sub => 2,
        BinaryOperation::Mul if lhs.last()?.min() > 0.0 || lhs.last()?.max() < 0.0 => 3,
        _ => return None,
    };
    let (Leaf::Noise(left), lhs_tail) = sampler_branch(lhs)? else {
        return None;
    };
    let (Leaf::Noise(right), rhs_tail) = sampler_branch(rhs)? else {
        return None;
    };
    Some(BinaryBranch {
        left: (left, lhs_tail),
        right: (right, rhs_tail),
        operation,
    })
}

struct SelectionBranch<'a> {
    selector: (&'a Noise, Tail),
    thresholds: &'static [f32],
    alternatives: Vec<(&'a Noise, Tail)>,
}

fn selection_branch<'a>(stack: &'a [Component<'_>]) -> Option<SelectionBranch<'a>> {
    let (Component::Dependent(Dep::IntervalSelect(node)), prefix) = stack.split_last()? else {
        return None;
    };
    if !(2..=MAX_ALTERNATIVES).contains(&node.functions_indices.len())
        || node.thresholds.len() + 1 != node.functions_indices.len()
    {
        return None;
    }
    let mut input = prefix.get(..=node.input_index)?;
    for _ in 0..8 {
        let (node, rest) = input.split_last()?;
        match node {
            Component::Chunk(ChunkSpecificNoiseFunctionComponent::Cache(cache)) => {
                input = rest.get(..=cache.input_index)?;
            }
            Component::PassThrough(pass) => input = rest.get(..=pass.input_index())?,
            _ => break,
        }
    }
    let (Leaf::Noise(selector), tail) = sampler_branch(input)? else {
        return None;
    };
    let mut alternatives = Vec::with_capacity(node.functions_indices.len());
    for &index in node.functions_indices {
        let (Leaf::Noise(noise), alt_tail) = sampler_branch(prefix.get(..=index)?)? else {
            return None;
        };
        alternatives.push((noise, alt_tail));
    }
    Some(SelectionBranch {
        selector: (selector, tail),
        thresholds: node.thresholds,
        alternatives,
    })
}

fn append_noise_layers(out: &mut Vec<VkLayer>, noise: &Noise) -> Option<(u32, u32)> {
    let layers = noise.sampler.layers();
    if layers.is_empty() || layers.len() > MAX_COLUMN_STATES {
        return None;
    }
    let offset = u32::try_from(out.len()).ok()?;
    for l in layers {
        out.push(VkLayer {
            offset_x: l.noise.offset_x,
            offset_y: l.noise.offset_y,
            offset_z: l.noise.offset_z,
            xz_scale: noise.data.xz_scale * l.frequency,
            y_scale: noise.data.y_scale * l.frequency,
            amplitude: l.amplitude,
            _pad0: 0,
            permutations: VkLayer::pack_permutations(l.noise.permutations()),
        });
    }
    let count = u32::try_from(layers.len()).ok()?;
    Some((offset, count))
}

fn append_blended_layers(
    out_layers: &mut Vec<VkLayer>,
    out_smears: &mut Vec<f64>,
    sampler: &InterpolatedNoiseSampler,
    tail: Tail,
) -> Option<BlendedDesc> {
    let stacks = sampler.volume_stacks();
    let main_count = stacks[0].0.layers.len();
    let min_count = stacks[1].0.layers.len();
    let total_count = main_count + min_count + stacks[2].0.layers.len();
    if main_count == 0
        || min_count == 0
        || total_count == main_count + min_count
        || total_count > MAX_COLUMN_STATES
    {
        return None;
    }
    let layer_offset = u32::try_from(out_layers.len()).ok()?;
    let smear_offset = u32::try_from(out_smears.len()).ok()?;
    for (stack, xz_scale, y_scale) in stacks {
        for layer in &stack.layers {
            let (noise, scale) = layer.noise.parts();
            if !scale.is_finite() || scale.abs() <= 1.0 / f64::from(i32::MAX) {
                return None;
            }
            out_layers.push(VkLayer {
                offset_x: noise.offset_x,
                offset_y: noise.offset_y,
                offset_z: noise.offset_z,
                xz_scale: xz_scale * layer.frequency,
                y_scale: y_scale * layer.frequency,
                amplitude: layer.amplitude,
                _pad0: 0,
                permutations: VkLayer::pack_permutations(noise.permutations()),
            });
            out_smears.push(scale);
        }
    }
    Some(BlendedDesc {
        layer_offset,
        smear_offset,
        main_count: main_count as u32,
        min_count: min_count as u32,
        total_count: total_count as u32,
        tail,
    })
}

fn grid_volume(stack: &[Component<'_>], target: &DensityVolume) -> Option<(DensityVolume, bool)> {
    let mut best: Option<(DensityVolume, usize, bool)> = None;
    for component in stack {
        let Component::Chunk(ChunkSpecificNoiseFunctionComponent::Interpolated(interp)) = component
        else {
            continue;
        };
        let Some(volume) = interp.input_volume(target) else {
            continue;
        };
        if !(256..=MAX_SAMPLES).contains(&volume.size()) || volume.size_z > 65535 {
            continue;
        }
        let translatable = interp.supports_chunk_translation();
        let entry = (volume, 1, translatable);
        match &mut best {
            Some((current, count, current_translatable)) if *current == volume => {
                *count += 1;
                *current_translatable &= translatable;
            }
            None => best = Some(entry),
            Some(_) => {}
        }
    }
    best.map(|(volume, _, translatable)| (volume, translatable))
}

const fn translate_volume(
    template: DensityVolume,
    target: &DensityVolume,
) -> Option<DensityVolume> {
    if template.step_block_x <= 0
        || template.step_block_y <= 0
        || template.step_block_z <= 0
        || target.min_block_y != template.min_block_y
    {
        return None;
    }
    let dx = target.min_block_x - template.min_block_x;
    let dz = target.min_block_z - template.min_block_z;
    if dx.rem_euclid(template.step_block_x) != 0 || dz.rem_euclid(template.step_block_z) != 0 {
        return None;
    }
    Some(DensityVolume {
        min_block_x: target.min_block_x,
        min_block_y: template.min_block_y,
        min_block_z: target.min_block_z,
        ..template
    })
}

#[expect(clippy::too_many_lines)]
fn build_plan(
    engine: &Engine,
    stack: &[Component<'_>],
    volume: DensityVolume,
) -> Option<Arc<Plan>> {
    let mut indices: Vec<(usize, Option<usize>)> = Vec::new();
    let mut raw_layers: Vec<VkLayer> = Vec::new();
    let mut raw_smears: Vec<f64> = Vec::new();
    let mut params = BatchParams::default();
    let mut plane_count = 0usize;
    let mut binary_count = 0usize;
    let mut selection_count = 0usize;
    let mut blended_entry: Option<(usize, BlendedDesc)> = None;

    let mut visited = vec![false; stack.len()];

    for idx in (0..stack.len()).rev() {
        if visited[idx] {
            continue;
        }
        let sub = &stack[..=idx];

        if selection_count < MAX_SELECTIONS
            && let Some(sel) = selection_branch(sub)
        {
            let total_cols: usize = sel.selector.0.sampler.layers().len()
                + sel
                    .alternatives
                    .iter()
                    .map(|(n, _)| n.sampler.layers().len())
                    .sum::<usize>();
            let needed_planes = 1 + sel.alternatives.len();
            if total_cols <= MAX_COLUMN_STATES && plane_count + needed_planes <= MAX_PLANES {
                let Some((sel_off, sel_cnt)) = append_noise_layers(&mut raw_layers, sel.selector.0)
                else {
                    continue;
                };
                let main_slot = plane_count;
                params.planes[main_slot] = PlaneDesc {
                    layer_offset: sel_off,
                    layer_count: sel_cnt,
                    tail: sel.selector.1,
                };
                plane_count += 1;

                let mut sel_desc = SelectorDesc {
                    threshold_count: sel.thresholds.len() as u32,
                    ..SelectorDesc::default()
                };
                for (i, &t) in sel.thresholds.iter().enumerate() {
                    sel_desc.thresholds[i] = t;
                }
                let mut ok = true;
                for (i, &(alt_noise, alt_tail)) in sel.alternatives.iter().enumerate() {
                    if let Some((alt_off, alt_cnt)) =
                        append_noise_layers(&mut raw_layers, alt_noise)
                    {
                        params.planes[plane_count] = PlaneDesc {
                            layer_offset: alt_off,
                            layer_count: alt_cnt,
                            tail: alt_tail,
                        };
                        sel_desc.alternatives[i] = plane_count as u32;
                        plane_count += 1;
                    } else {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    params.selectors[selection_count] = sel_desc;
                    selection_count += 1;
                    params.selector_slots[main_slot] = selection_count as u32;
                    indices.push((idx, Some(main_slot)));
                    mark_children(sub, &mut visited);
                    continue;
                }
            }
        }

        if binary_count < MAX_BINARY_PAIRS
            && plane_count < MAX_PLANES
            && let Some(BinaryBranch {
                left: (left, lhs_tail),
                right: (right, rhs_tail),
                operation: op,
            }) = binary_branch(sub)
        {
            let total_cols = left.sampler.layers().len() + right.sampler.layers().len();
            if total_cols <= MAX_COLUMN_STATES
                && let Some((l_off, l_cnt)) = append_noise_layers(&mut raw_layers, left)
                && let Some((r_off, r_cnt)) = append_noise_layers(&mut raw_layers, right)
            {
                let slot = plane_count;
                params.planes[slot] = PlaneDesc {
                    layer_offset: l_off,
                    layer_count: l_cnt,
                    tail: lhs_tail,
                };
                params.companions[binary_count] = PlaneDesc {
                    layer_offset: r_off,
                    layer_count: r_cnt,
                    tail: rhs_tail,
                };
                params.binary_ops[binary_count] = op;
                binary_count += 1;
                params.binary_slots[slot] = binary_count as u32;
                plane_count += 1;
                indices.push((idx, Some(slot)));
                mark_children(sub, &mut visited);
                continue;
            }
        }

        if let Some((leaf, tail)) = sampler_branch(sub) {
            match leaf {
                Leaf::Noise(noise) if plane_count < MAX_PLANES => {
                    if let Some((off, cnt)) = append_noise_layers(&mut raw_layers, noise) {
                        let slot = plane_count;
                        params.planes[slot] = PlaneDesc {
                            layer_offset: off,
                            layer_count: cnt,
                            tail,
                        };
                        plane_count += 1;
                        indices.push((idx, Some(slot)));
                        mark_children(sub, &mut visited);
                    }
                }
                Leaf::InterpolatedNoise(blended) if blended_entry.is_none() => {
                    if let Some(desc) =
                        append_blended_layers(&mut raw_layers, &mut raw_smears, blended, tail)
                    {
                        blended_entry = Some((idx, desc));
                        mark_children(sub, &mut visited);
                    }
                }
                _ => {}
            }
        }
    }

    let total_planes = plane_count + usize::from(blended_entry.is_some());
    if total_planes == 0 {
        return None;
    }
    if let Some((idx, desc)) = blended_entry {
        params.blended = desc;
        indices.push((idx, Some(total_planes - 1)));
    }
    if raw_layers.is_empty() {
        return None;
    }
    if raw_smears.is_empty() {
        raw_smears.push(0.0);
    }

    let layers = engine.device.upload_storage_slice(&raw_layers).ok()?;
    let smears = engine.device.upload_storage_slice(&raw_smears).ok()?;

    params.plane_count = total_planes as u32;
    params.size_x = volume.size_x as i32;
    params.size_y = volume.size_y as i32;
    params.size_z = volume.size_z as i32;
    params.step_x = volume.step_block_x;
    params.step_y = volume.step_block_y;
    params.step_z = volume.step_block_z;

    Some(Arc::new(Plan {
        volume,
        indices,
        plane_count: total_planes,
        layers,
        smears,
        base_params: params,
    }))
}

fn mark_children(sub: &[Component<'_>], visited: &mut [bool]) {
    let Some((node, _)) = sub.split_last() else {
        return;
    };
    let idx = sub.len() - 1;
    if visited[idx] {
        return;
    }
    visited[idx] = true;
    match node {
        Component::PassThrough(p) => mark_children(&sub[..=p.input_index()], visited),
        Component::Dependent(Dep::Linear(n)) => mark_children(&sub[..=n.input_index], visited),
        Component::Dependent(Dep::Clamp(n)) => mark_children(&sub[..=n.input_index], visited),
        Component::Dependent(Dep::Unary(n)) => mark_children(&sub[..=n.input_index], visited),
        Component::Dependent(Dep::Binary(n)) => {
            mark_children(&sub[..=n.input1_index], visited);
            mark_children(&sub[..=n.input2_index], visited);
        }
        Component::Dependent(Dep::IntervalSelect(n)) => {
            mark_children(&sub[..=n.input_index], visited);
            for &alt in n.functions_indices {
                mark_children(&sub[..=alt], visited);
            }
        }
        Component::Chunk(ChunkSpecificNoiseFunctionComponent::Cache(c)) => {
            mark_children(&sub[..=c.input_index], visited);
        }
        _ => {}
    }
}

fn prepare_cached_plan(engine: &Engine, generator: &VanillaGenerator) -> Option<Arc<Plan>> {
    let target = generator.noise_density_volume(0, 0);
    let router = ChunkNoiseRouter::generate_for_prefetch(&generator.base_router.noise);
    let stack = router.final_density_stack();
    let (volume, translatable) = grid_volume(stack, &target)?;
    if !translatable {
        return None;
    }
    build_plan(engine, stack, volume)
}

fn enqueue(
    engine: &Engine,
    plan: Arc<Plan>,
    volume: DensityVolume,
    base: usize,
    cancel: Option<Arc<AtomicBool>>,
) -> Option<Prefetched> {
    let (reply, rx) = bounded(1);
    let job = Job {
        volume,
        plan: Arc::clone(&plan),
        reply,
        cancel,
    };
    if !submit(engine, job) {
        return None;
    }
    Some(Prefetched {
        base,
        volume,
        plan,
        result: ResultState::new(rx),
        used: Cell::new(0),
    })
}

pub fn prepare_chunk(
    generator: &VanillaGenerator,
    chunk_x: i32,
    chunk_z: i32,
    cancel: Option<Arc<AtomicBool>>,
) -> Option<PreparedGrid> {
    let engine = engine()?;
    let target = generator.noise_density_volume(chunk_x, chunk_z);
    let cached_plan = generator
        .gpu_plan
        .vulkan
        .get_or_init(|| prepare_cached_plan(engine, generator))
        .as_ref()?;
    let volume = translate_volume(cached_plan.volume, &target)?;
    let batch = enqueue(engine, Arc::clone(cached_plan), volume, 0, cancel)?;
    Some(PreparedGrid { target, batch })
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

pub fn try_sample_density(
    stack: &mut [Component<'_>],
    out: &mut [f32],
    target: &DensityVolume,
    source: &ProtoNoiseRouter,
) -> bool {
    if CHECKING_CPU.get() || PREFETCHED.with_borrow(Option::is_some) {
        return false;
    }
    let Some(engine) = engine() else {
        return false;
    };
    let stack_base = stack.as_ptr() as usize;
    let prefetched = EARLY_GRID
        .with_borrow_mut(|early| {
            early
                .as_mut()
                .and_then(|entry| entry.bind(source, target, stack_base))
        })
        .or_else(|| {
            let (volume, _) = grid_volume(stack, target)?;
            let plan = build_plan(engine, stack, volume)?;
            enqueue(engine, plan, volume, stack_base, None)
        });
    let Some(batch) = prefetched else {
        return false;
    };
    PREFETCHED.set(Some(batch));
    let _clear = ClearPrefetched;
    Component::sample_volume_from_stack(stack, out, target);
    true
}

pub fn try_prefetched(stack: &[Component<'_>], out: &mut [f32], volume: &DensityVolume) -> bool {
    if CHECKING_CPU.get() || out.len() != volume.size() {
        return false;
    }
    PREFETCHED.with_borrow(|prefetched| {
        let Some(batch) = prefetched else {
            return false;
        };
        if batch.volume != *volume {
            return false;
        }
        let base = stack.as_ptr() as usize;
        if base != batch.base {
            return false;
        }
        let index = stack.len().saturating_sub(1);
        let Some((_, Some(slot))) = batch
            .plan
            .indices
            .iter()
            .find(|&&(node_idx, _)| node_idx == index)
        else {
            return false;
        };
        let mask = 1u32.checked_shl(*slot as u32).unwrap_or(0);
        if (batch.used.get() & mask) != 0 {
            return false;
        }
        let Some(readback) = batch.result.get() else {
            return false;
        };
        let len = volume.size();
        let start = readback.offset + *slot * len;
        let Some(slice) = readback.values.get(start..start + len) else {
            return false;
        };
        out.copy_from_slice(slice);
        batch.used.set(batch.used.get() | mask);
        super::HITS.fetch_add(1, Ordering::Relaxed);
        true
    })
}

use super::{
    Job, MAX_BATCH, MAX_SAMPLES, Request, Tail, blended, disable, fusion, grid, selection,
    smear_table,
};
use crossbeam::channel::Receiver;
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, DriverError, LaunchConfig,
    PinnedHostSlice, PushKernelArg,
};
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[repr(C)]
struct Origins([i32; MAX_BATCH * 3]);
// SAFETY: Contiguous initialized coordinates passed by value, as in worldgen.cu.
unsafe impl DeviceRepr for Origins {}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct PlaneDescriptor {
    layers: u64,
    layer_count: u32,
    tail: Tail,
}

#[repr(C)]
struct GridDescriptor([PlaneDescriptor; grid::MAX_PLANES]);
const _: () = assert!(std::mem::size_of::<GridDescriptor>() == 2800);

#[derive(Default)]
#[repr(C)]
struct BinaryOperands {
    companions: [PlaneDescriptor; fusion::MAX_PAIRS],
    operations: [u32; fusion::MAX_PAIRS],
    slots: [u32; grid::MAX_PLANES],
}

const _: () = assert!(std::mem::size_of::<BinaryOperands>() == 568);
// SAFETY: Matches the initialized scalar descriptors in worldgen.cu.
unsafe impl DeviceRepr for BinaryOperands {}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Selector {
    threshold_count: u32,
    thresholds: [f32; selection::MAX_ALTERNATIVES - 1],
    alternatives: [u32; selection::MAX_ALTERNATIVES],
}

#[derive(Default)]
#[repr(C)]
struct Selectors {
    entries: [Selector; selection::MAX_SELECTIONS],
    slots: [u32; grid::MAX_PLANES],
}

const _: () = assert!(std::mem::size_of::<Selectors>() == 220);
// SAFETY: Matches the initialized scalar descriptors in worldgen.cu.
unsafe impl DeviceRepr for Selectors {}

use smear_table::MAX_GRID_CHUNKS;
const COLUMN_STATE_BYTES: u32 = 28;
const DEFAULT_SHARED_MEMORY_BYTES: u32 = 48 * 1024;
pub(super) const MAX_COLUMN_STATES: usize =
    (DEFAULT_SHARED_MEMORY_BYTES / COLUMN_STATE_BYTES) as usize;

#[repr(C)]
struct GridOrigins([i32; MAX_GRID_CHUNKS * 3]);
// SAFETY: CUDA receives the same initialized array of coordinates by value.
unsafe impl DeviceRepr for GridOrigins {}

const _: () = assert!(std::mem::size_of::<PlaneDescriptor>() == 112);
// SAFETY: The descriptor has the same initialized scalar layout as GridDescriptor in CUDA.
unsafe impl DeviceRepr for GridDescriptor {}

pub(super) struct Lane {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    output: Option<(CudaSlice<f32>, PinnedHostSlice<f32>)>,
    terrain_pair: bool,
    fuse_density: bool,
    select_density: bool,
    smear_table: Option<smear_table::Cache>,
}

impl Lane {
    pub(super) const fn new(
        stream: Arc<CudaStream>,
        kernel: CudaFunction,
        terrain_pair: bool,
        fuse_density: bool,
    ) -> Self {
        Self {
            stream,
            kernel,
            output: None,
            terrain_pair,
            fuse_density,
            select_density: false,
            smear_table: None,
        }
    }

    pub(super) const fn enable_selection(&mut self) {
        self.select_density = true;
    }

    pub(super) fn enable_smear_table(&mut self, prepare: CudaFunction) {
        self.smear_table = Some(smear_table::Cache::new(prepare));
    }

    pub(super) fn run(mut self, rx: &Receiver<Request>, disabled: &AtomicBool) {
        let mut pending = Vec::with_capacity(MAX_BATCH);
        let mut batch = Vec::with_capacity(MAX_BATCH);
        while let Ok(Request::Volume(first)) = rx.recv() {
            pending.push(first);
            pending.extend(
                rx.try_iter()
                    .take(MAX_BATCH - 1)
                    .filter_map(|request| match request {
                        Request::Volume(job) => Some(job),
                        Request::Grid(_) => None,
                    }),
            );
            while let Some(first) = pending.pop() {
                if disabled.load(Ordering::Acquire) {
                    pending.clear();
                    break;
                }
                let cap = MAX_SAMPLES / first.volume.size();
                batch.extend(
                    pending
                        .extract_if(.., |job| compatible(&first, job))
                        .take(cap - 1),
                );
                batch.push(first);
                if let Err(error) = self.dispatch(&batch) {
                    disable(disabled, error);
                }
                // Dropping the sender wakes every caller, including on CUDA errors.
                batch.clear();
            }
        }
    }

    pub(super) fn run_grids(
        mut self,
        rx: &Receiver<Request>,
        disabled: &AtomicBool,
        pipeline: bool,
        terrain: bool,
    ) {
        let mut pending = VecDeque::with_capacity(MAX_GRID_CHUNKS);
        let mut batch = Vec::with_capacity(MAX_GRID_CHUNKS);
        loop {
            let first = match pending.pop_front() {
                Some(job) => job,
                None => match rx.recv() {
                    Ok(Request::Grid(job)) => job,
                    _ => break,
                },
            };
            if disabled.load(Ordering::Acquire) {
                pending.clear();
                continue;
            }
            if first.is_cancelled() {
                continue;
            }
            if pipeline {
                pending.extend(
                    rx.try_iter()
                        .take(MAX_GRID_CHUNKS - 1 - pending.len())
                        .filter_map(|request| match request {
                            Request::Grid(job) => Some(job),
                            Request::Volume(_) => None,
                        }),
                );
                pending.retain(|job| !job.is_cancelled());
                let limit = (MAX_SAMPLES / (first.plane_count() * first.volume.size()))
                    .min(MAX_GRID_CHUNKS);
                while batch.len() + 1 < limit {
                    let Some(index) = pending.iter().position(|job| compatible_grid(&first, job))
                    else {
                        break;
                    };
                    if let Some(job) = pending.remove(index) {
                        batch.push(job);
                    }
                }
            }
            // Index zero supplies the canonical sampler descriptors for the launch.
            batch.insert(0, first);
            if let Err(error) = self.dispatch_grid(&batch, pipeline, terrain) {
                disable(disabled, error);
            }
            batch.clear();
        }
    }

    fn reserve_output(&mut self, len: usize) -> Result<(), DriverError> {
        if self
            .output
            .as_ref()
            .is_none_or(|(device, _)| device.len() < len)
        {
            let capacity = len.next_power_of_two();
            let device = self.stream.alloc_zeros(capacity)?;
            // SAFETY: Initialize the complete pinned allocation before exposing typed slices.
            let host = unsafe {
                let mut host = self
                    .stream
                    .context()
                    .alloc_pinned_with_flags::<f32>(capacity, 0)?;
                host.as_mut_ptr()?.write_bytes(0, capacity);
                host
            };
            self.output = Some((device, host));
        }
        Ok(())
    }

    #[expect(clippy::too_many_lines)]
    pub(super) fn dispatch_grid(
        &mut self,
        batch: &[grid::Job],
        pipeline: bool,
        terrain: bool,
    ) -> Result<(), DriverError> {
        let job = &batch[0];
        let v = job.volume;
        let samples_per_chunk = v.size() * job.plane_count();
        let len = samples_per_chunk * batch.len();
        self.reserve_output(len)?;
        let Some((device, host)) = &mut self.output else {
            return Ok(());
        };
        let mut grid = GridDescriptor([PlaneDescriptor::default(); grid::MAX_PLANES]);
        let mut binary = BinaryOperands::default();
        let mut binary_count = 0;
        let mut selectors = Selectors::default();
        let mut selected_count = 0;
        // Alternatives are sampler descriptors only. The planner counts each one
        // against MAX_PLANES, so they fit after the output plane descriptors.
        let mut spare_slot = job.plan.planes.len();
        let mut reads =
            Vec::with_capacity(job.plane_count() + usize::from(job.plan.blended.is_some()));
        let mut max_layers = 0;
        for (index, plane) in job.plan.planes.iter().enumerate() {
            let (layers, record_read) = plane.layers.device_ptr(&self.stream);
            reads.push(record_read);
            let layer_count = plane.layers.len() as u32;
            max_layers = max_layers.max(layer_count);
            grid.0[index] = PlaneDescriptor {
                layers,
                layer_count,
                tail: plane.tail,
            };
            if let Some(companion) = &plane.companion {
                let (layers, record_read) = companion.layers.device_ptr(&self.stream);
                reads.push(record_read);
                let right_count = companion.layers.len() as u32;
                max_layers = max_layers.max(layer_count + right_count);
                binary.companions[binary_count] = PlaneDescriptor {
                    layers,
                    layer_count: right_count,
                    tail: companion.tail,
                };
                binary.operations[binary_count] = companion.operation as u32;
                binary_count += 1;
                binary.slots[index] = binary_count as u32;
            }
            if let Some(selection) = &plane.selection {
                let selector = &mut selectors.entries[selected_count];
                selector.threshold_count = selection.thresholds.len() as u32;
                selector.thresholds[..selection.thresholds.len()]
                    .copy_from_slice(selection.thresholds);
                let mut total_layers = layer_count;
                for (slot, alternative) in selector
                    .alternatives
                    .iter_mut()
                    .zip(&selection.alternatives)
                {
                    let (layers, record_read) = alternative.layers.device_ptr(&self.stream);
                    reads.push(record_read);
                    let layer_count = alternative.layers.len() as u32;
                    total_layers += layer_count;
                    grid.0[spare_slot] = PlaneDescriptor {
                        layers,
                        layer_count,
                        tail: alternative.tail,
                    };
                    *slot = spare_slot as u32;
                    spare_slot += 1;
                }
                max_layers = max_layers.max(total_layers);
                selected_count += 1;
                selectors.slots[index] = selected_count as u32;
            }
        }
        let blend = if let Some(plane) = &job.plan.blended {
            let (layers, record_layers) = plane.layers.layers.device_ptr(&self.stream);
            let (smear, record_smear) = plane.layers.smear.device_ptr(&self.stream);
            reads.push(record_layers);
            reads.push(record_smear);
            let total = plane.layers.layers.len() as u32;
            max_layers = max_layers.max(total);
            blended::Descriptor {
                layers,
                smear,
                tail: plane.tail,
                main: plane.layers.main,
                min: plane.layers.min,
                total,
            }
        } else {
            blended::Descriptor::default()
        };
        let mut origins = GridOrigins([0; MAX_GRID_CHUNKS * 3]);
        for (coords, item) in origins.0.as_chunks_mut::<3>().0.iter_mut().zip(batch) {
            *coords = [
                item.volume.min_block_x,
                item.volume.min_block_y,
                item.volume.min_block_z,
            ];
        }
        let pair_columns = self.terrain_pair
            && v.size_x > 1
            && blend.total > 0
            && 2 * blend.total * COLUMN_STATE_BYTES <= DEFAULT_SHARED_MEMORY_BYTES;
        if pair_columns {
            max_layers = max_layers.max(2 * blend.total);
        }
        let pair_columns_arg = u32::from(pair_columns);
        let plane_count = job.plane_count() as u32;
        let total_planes = job.plane_count() * batch.len();
        let sx = v.size_x as i32;
        let sy = v.size_y as i32;
        let sz = v.size_z as i32;
        let config = LaunchConfig {
            grid_dim: (sx as u32, sz as u32, total_planes as u32),
            block_dim: ((sy as u32).next_multiple_of(32).min(256), 1, 1),
            shared_mem_bytes: max_layers * COLUMN_STATE_BYTES,
        };
        let uses_table_kernel = self.smear_table.is_some();
        let table = match &mut self.smear_table {
            Some(cache) => cache.prepare_for(batch, &blend, &self.stream)?,
            None => None,
        };
        let used_table = table.is_some();
        let built_table = table.as_ref().is_some_and(|table| table.built);
        let null_table = 0u64;
        let empty_rows = smear_table::TableRows::default();
        let mut output = device.slice_mut(..len);
        let mut launch = self.stream.launch_builder(&self.kernel);
        launch.arg(&grid);
        if terrain {
            launch.arg(&blend);
        }
        if pipeline {
            launch.arg(&origins);
            launch.arg(&plane_count);
        } else {
            launch.arg(&origins.0[0]);
            launch.arg(&origins.0[1]);
            launch.arg(&origins.0[2]);
        }
        launch.arg(&sx);
        launch.arg(&sy);
        launch.arg(&sz);
        launch.arg(&v.step_block_x);
        launch.arg(&v.step_block_y);
        launch.arg(&v.step_block_z);
        if self.terrain_pair {
            launch.arg(&pair_columns_arg);
        }
        if uses_table_kernel {
            if let Some(table) = &table {
                launch.arg(&table.rows);
                launch.arg(table.buffer);
            } else {
                launch.arg(&empty_rows);
                launch.arg(&null_table);
            }
        }
        if self.fuse_density {
            launch.arg(&binary);
        }
        if self.select_density {
            launch.arg(&selectors);
        }
        launch.arg(&mut output);
        // SAFETY: The descriptors and layer allocations are live, the grid indexes one
        // descriptor per plane, and the output covers all planes in the same layout.
        unsafe { launch.launch(config) }?;
        drop(reads);

        let result = &mut host.as_mut_slice()?[..len];
        self.stream.memcpy_dtoh(&output, result)?;
        self.stream.synchronize()?;
        super::READBACK_BYTES.fetch_add(std::mem::size_of_val(result) as u64, Ordering::Relaxed);
        super::BATCHES.fetch_add(1 + u64::from(built_table), Ordering::Relaxed);
        if used_table {
            super::SMEAR_TABLE_BATCHES.fetch_add(1, Ordering::Relaxed);
        }
        if built_table {
            super::SMEAR_TABLE_BUILDS.fetch_add(1, Ordering::Relaxed);
        }
        super::GRID_BATCHES.fetch_add(1, Ordering::Relaxed);
        if pair_columns {
            super::PAIRED_BLENDED_BATCHES.fetch_add(1, Ordering::Relaxed);
        }
        super::VOLUMES.fetch_add(total_planes as u64, Ordering::Relaxed);
        super::GRID_CHUNKS.fetch_add(batch.len() as u64, Ordering::Relaxed);
        let fused = job
            .plan
            .planes
            .iter()
            .filter(|plane| {
                plane.tail.count > 0 || plane.companion.is_some() || plane.selection.is_some()
            })
            .count()
            + usize::from(
                job.plan
                    .blended
                    .as_ref()
                    .is_some_and(|plane| plane.tail.count > 0),
            );
        super::FUSED.fetch_add((fused * batch.len()) as u64, Ordering::Relaxed);
        let values: Arc<[f32]> = Arc::from(result as &[f32]);
        for (index, item) in batch.iter().enumerate() {
            let _ = item.reply.send(grid::Readback {
                values: values.clone(),
                offset: index * samples_per_chunk,
            });
        }
        Ok(())
    }

    fn dispatch(&mut self, batch: &[Job]) -> Result<(), DriverError> {
        let first = &batch[0];
        let v = first.volume;
        let len = v.size() * batch.len();
        self.reserve_output(len)?;
        let Some((device, host)) = &mut self.output else {
            return Ok(());
        };
        let mut origins = Origins([0; MAX_BATCH * 3]);
        for (coords, job) in origins.0.as_chunks_mut::<3>().0.iter_mut().zip(batch) {
            *coords = [
                job.volume.min_block_x,
                job.volume.min_block_y,
                job.volume.min_block_z,
            ];
        }

        let layers = first.layers.len() as u32;
        let sx = v.size_x as i32;
        let sy = v.size_y as i32;
        let sz = v.size_z as i32;
        let config = LaunchConfig {
            grid_dim: (sx as u32, sz as u32, batch.len() as u32),
            block_dim: ((sy as u32).next_multiple_of(32).min(256), 1, 1),
            shared_mem_bytes: layers * COLUMN_STATE_BYTES,
        };
        let mut output = device.slice_mut(..len);
        let mut launch = self.stream.launch_builder(&self.kernel);
        launch.arg(first.layers.as_ref());
        launch.arg(&layers);
        launch.arg(&origins);
        launch.arg(&sx);
        launch.arg(&sy);
        launch.arg(&sz);
        launch.arg(&v.step_block_x);
        launch.arg(&v.step_block_y);
        launch.arg(&v.step_block_z);
        launch.arg(&first.tail);
        launch.arg(&mut output);
        // SAFETY: All jobs share the geometry, sampler and operation tail. The grid has
        // one block per X/Z column per job, and output spans the entire batch.
        unsafe { launch.launch(config) }?;
        // Copy only the valid prefix of the reusable pinned allocation. This lane
        // owns it until its stream finishes; the other lane continues independently.
        let result = &mut host.as_mut_slice()?[..len];
        self.stream.memcpy_dtoh(&output, result)?;
        self.stream.synchronize()?;
        super::READBACK_BYTES.fetch_add(std::mem::size_of_val(result) as u64, Ordering::Relaxed);
        super::BATCHES.fetch_add(1, Ordering::Relaxed);
        super::VOLUMES.fetch_add(batch.len() as u64, Ordering::Relaxed);
        if first.tail.count > 0 {
            super::FUSED.fetch_add(batch.len() as u64, Ordering::Relaxed);
        }
        for (job, values) in batch.iter().zip(result.chunks_exact(v.size())) {
            let _ = job.reply.send(values.to_vec());
        }
        Ok(())
    }
}

fn compatible_grid(a: &grid::Job, b: &grid::Job) -> bool {
    same_geometry(&a.volume, &b.volume)
        && (Arc::ptr_eq(&a.plan, &b.plan) || compatible_plans(&a.plan, &b.plan))
}

fn compatible_plans(a: &grid::Plan, b: &grid::Plan) -> bool {
    a.planes.len() == b.planes.len()
        && match (&a.blended, &b.blended) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(&a.layers, &b.layers) && a.tail == b.tail,
            _ => false,
        }
        && a.planes.iter().zip(&b.planes).all(|(a, b)| {
            Arc::ptr_eq(&a.layers, &b.layers)
                && a.tail == b.tail
                && match (&a.companion, &b.companion) {
                    (None, None) => true,
                    (Some(a), Some(b)) => {
                        Arc::ptr_eq(&a.layers, &b.layers)
                            && a.tail == b.tail
                            && a.operation == b.operation
                    }
                    _ => false,
                }
                && match (&a.selection, &b.selection) {
                    (None, None) => true,
                    (Some(a), Some(b)) => {
                        std::ptr::eq(a.thresholds, b.thresholds)
                            && a.alternatives.len() == b.alternatives.len()
                            && a.alternatives
                                .iter()
                                .zip(&b.alternatives)
                                .all(|(a, b)| Arc::ptr_eq(&a.layers, &b.layers) && a.tail == b.tail)
                    }
                    _ => false,
                }
        })
}

fn compatible(a: &Job, b: &Job) -> bool {
    Arc::ptr_eq(&a.layers, &b.layers) && a.tail == b.tail && same_geometry(&a.volume, &b.volume)
}

fn same_geometry(
    x: &pumpkin_util::noise::volume::DensityVolume,
    y: &pumpkin_util::noise::volume::DensityVolume,
) -> bool {
    (
        x.size_x,
        x.size_y,
        x.size_z,
        x.step_block_x,
        x.step_block_y,
        x.step_block_z,
    ) == (
        y.size_x,
        y.size_y,
        y.size_z,
        y.step_block_x,
        y.step_block_y,
        y.step_block_z,
    )
}

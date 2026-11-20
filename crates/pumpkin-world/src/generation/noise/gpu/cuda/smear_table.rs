use std::sync::Arc;

use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DeviceRepr, DriverError, LaunchConfig, PushKernelArg,
    ValidAsZeroBits,
};

use super::{blended, grid};

pub(super) const MAX_GRID_CHUNKS: usize = 8;
const MAX_TABLE_ENTRIES: usize = 1 << 20;
const PREPARE_THREADS: u32 = 128;

#[repr(C)]
pub(super) struct SmearY {
    floor_y: i32,
    fudged_y: f32,
    alpha: f32,
}

const _: () = assert!(std::mem::size_of::<SmearY>() == 12);
// SAFETY: Matches SmearY in worldgen.cu and contains only initialized scalars.
unsafe impl DeviceRepr for SmearY {}
// SAFETY: Zero bits are valid for the integer and both floats.
unsafe impl ValidAsZeroBits for SmearY {}

#[derive(Default)]
#[repr(C)]
pub(super) struct TableRows([u32; MAX_GRID_CHUNKS]);
// SAFETY: CUDA receives the same initialized array of indices by value.
unsafe impl DeviceRepr for TableRows {}

#[repr(C)]
struct BaseYs([i32; MAX_GRID_CHUNKS]);
// SAFETY: CUDA receives the same initialized array of coordinates by value.
unsafe impl DeviceRepr for BaseYs {}

struct Key {
    sampler: Arc<blended::BlendedLayers>,
    size_y: usize,
    step_y: i32,
    base_y: Vec<i32>,
}

pub(super) struct Cache {
    prepare: CudaFunction,
    buffer: Option<CudaSlice<SmearY>>,
    key: Option<Key>,
}

pub(super) struct Table<'a> {
    pub buffer: &'a CudaSlice<SmearY>,
    pub rows: TableRows,
    pub built: bool,
}

impl Cache {
    pub(super) const fn new(prepare: CudaFunction) -> Self {
        Self {
            prepare,
            buffer: None,
            key: None,
        }
    }

    pub(super) fn prepare_for<'a>(
        &'a mut self,
        batch: &[grid::Job],
        descriptor: &blended::Descriptor,
        stream: &Arc<CudaStream>,
    ) -> Result<Option<Table<'a>>, DriverError> {
        let first = &batch[0];
        let Some(blend) = &first.plan.blended else {
            return Ok(None);
        };
        let v = first.volume;
        if let Some(key) = &self.key
            && Arc::ptr_eq(&blend.layers, &key.sampler)
            && key.size_y == v.size_y
            && key.step_y == v.step_block_y
            && let Some(rows) = rows_for(batch, &key.base_y)
        {
            return Ok(self.buffer.as_ref().map(|buffer| Table {
                buffer,
                rows,
                built: false,
            }));
        }

        let mut bases = Vec::with_capacity(batch.len());
        for job in batch {
            let base = job.volume.min_block_y;
            if !bases.contains(&base) {
                bases.push(base);
            }
        }
        let Some(rows) = rows_for(batch, &bases) else {
            return Ok(None);
        };
        let row_entries = descriptor.total as usize * v.size_y;
        let entries = row_entries * bases.len();
        if entries == 0 || entries > MAX_TABLE_ENTRIES {
            return Ok(None);
        }
        if self.buffer.as_ref().is_none_or(|b| b.len() < entries) {
            self.buffer = Some(stream.alloc_zeros(entries.next_power_of_two())?);
        }
        let mut origins = BaseYs([0; MAX_GRID_CHUNKS]);
        origins.0[..bases.len()].copy_from_slice(&bases);
        let size_y = v.size_y as i32;
        let row_entries = row_entries as u32;
        let Some(buffer) = &mut self.buffer else {
            return Ok(None);
        };
        let mut launch = stream.launch_builder(&self.prepare);
        launch.arg(descriptor);
        launch.arg(&origins);
        launch.arg(&size_y);
        launch.arg(&v.step_block_y);
        launch.arg(&row_entries);
        launch.arg(&mut *buffer);
        // SAFETY: One disjoint table row per origin. Every entry for each row is written,
        // and the caller holds the sampler allocations on this stream until rendering finishes.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (row_entries.div_ceil(PREPARE_THREADS), bases.len() as u32, 1),
                block_dim: (PREPARE_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        self.key = Some(Key {
            sampler: blend.layers.clone(),
            size_y: v.size_y,
            step_y: v.step_block_y,
            base_y: bases,
        });
        Ok(Some(Table {
            buffer,
            rows,
            built: true,
        }))
    }
}

fn rows_for(batch: &[grid::Job], bases: &[i32]) -> Option<TableRows> {
    let mut rows = TableRows::default();
    for (row, job) in rows.0.iter_mut().zip(batch) {
        *row = bases
            .iter()
            .position(|&base| base == job.volume.min_block_y)? as u32;
    }
    Some(rows)
}

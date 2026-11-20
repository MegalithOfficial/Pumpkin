//! Safe `vulkano` compute device, buffer, and worker lane abstraction.

use super::abi::{BatchParams, MAX_GRID_CHUNKS, PrepareParams, SmearY};
use std::sync::Arc;
use vulkano::{
    VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
        allocator::{
            CommandBufferAllocator, StandardCommandBufferAllocator,
            StandardCommandBufferAllocatorCreateInfo,
        },
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        allocator::{
            DescriptorSetAllocator, StandardDescriptorSetAllocator,
            StandardDescriptorSetAllocatorCreateInfo,
        },
    },
    device::{Device, DeviceCreateInfo, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags},
    instance::{Instance, InstanceCreateInfo},
    memory::{
        MemoryHeapFlags,
        allocator::{
            AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
        },
    },
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
    sync::{self, GpuFuture},
};

const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const WORKGROUP_SIZE: usize = 64;
const PREPARE_WORKGROUP_SIZE: u32 = 128;
const MAX_SMEAR_ENTRIES: usize = 1 << 20;
const MAX_LANES: usize = 8;

pub(super) type StorageBuffer<T> = Subbuffer<[T]>;

pub(super) struct DeviceInfo {
    pub name: String,
    pub api_version: String,
    pub vram_bytes: u64,
}

pub(super) struct VulkanDevice {
    device: Arc<Device>,
    queues: Vec<Arc<Queue>>,
    queue_family: u32,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_allocator: Arc<dyn CommandBufferAllocator>,
    descriptor_allocator: Arc<dyn DescriptorSetAllocator>,
    pipeline: Arc<ComputePipeline>,
    prepare_pipeline: Arc<ComputePipeline>,
    pub info: DeviceInfo,
}

impl VulkanDevice {
    #[expect(clippy::too_many_lines)]
    pub(super) fn create(
        device_index: usize,
        configured_streams: usize,
    ) -> Result<(Arc<Self>, usize, &'static str), String> {
        let library =
            VulkanLibrary::new().map_err(|_| String::from("Vulkan loader library not found"))?;
        let instance = Instance::new(library, InstanceCreateInfo::default())
            .map_err(|e| format!("failed to create Vulkan instance: {e}"))?;

        let physical_devices: Vec<_> = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("failed to enumerate Vulkan devices: {e}"))?
            .collect();
        let count = physical_devices.len();
        if count == 0 {
            return Err(String::from("no Vulkan-capable GPU detected"));
        }
        if device_index >= count {
            return Err(if count == 1 {
                format!("vulkan:{device_index} not found (only device 0 is available)")
            } else {
                format!("vulkan:{device_index} not found (available devices: 0..{count})")
            });
        }

        let physical = physical_devices[device_index].clone();
        let props = physical.properties();
        let name = props.device_name.clone();
        let api_version = format!(
            "{}.{}.{}",
            props.api_version.major, props.api_version.minor, props.api_version.patch
        );

        if !physical.supported_features().shader_float64 {
            return Err(format!(
                "vulkan:{device_index} ({name}) does not support shaderFloat64"
            ));
        }

        let vram_bytes = physical
            .memory_properties()
            .memory_heaps
            .iter()
            .filter(|heap| heap.flags.contains(MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .sum::<u64>();

        let queue_families = physical.queue_family_properties();
        let queue_family = queue_families
            .iter()
            .enumerate()
            .find(|(_, q)| {
                q.queue_flags.contains(QueueFlags::COMPUTE)
                    && !q.queue_flags.contains(QueueFlags::GRAPHICS)
                    && q.queue_count > 0
            })
            .or_else(|| {
                queue_families
                    .iter()
                    .enumerate()
                    .find(|(_, q)| q.queue_flags.contains(QueueFlags::COMPUTE) && q.queue_count > 0)
            })
            .map(|(idx, q)| (idx as u32, q.queue_count as usize));

        let Some((queue_family, available_queues)) = queue_family else {
            return Err(format!(
                "vulkan:{device_index} ({name}) has no compute queue family"
            ));
        };

        let host_cpus = std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get);
        let (lanes, source) = if configured_streams > 0 {
            (configured_streams.min(MAX_LANES), "config")
        } else {
            let auto = 2usize.min((host_cpus / 2).max(1));
            (auto, "auto")
        };

        let queue_count = lanes.min(available_queues).max(1);
        let (device, queues) = Device::new(
            physical,
            DeviceCreateInfo {
                enabled_features: DeviceFeatures {
                    shader_float64: true,
                    ..DeviceFeatures::empty()
                },
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: queue_family,
                    queues: vec![1.0; queue_count],
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .map_err(|e| {
            format!("failed to create Vulkan logical device on vulkan:{device_index}: {e}")
        })?;

        let queues: Vec<_> = queues.collect();
        let memory_allocator: Arc<dyn MemoryAllocator> =
            Arc::new(StandardMemoryAllocator::new_default(Arc::clone(&device)));
        let command_allocator: Arc<dyn CommandBufferAllocator> =
            Arc::new(StandardCommandBufferAllocator::new(
                Arc::clone(&device),
                StandardCommandBufferAllocatorCreateInfo::default(),
            ));
        let descriptor_allocator: Arc<dyn DescriptorSetAllocator> =
            Arc::new(StandardDescriptorSetAllocator::new(
                Arc::clone(&device),
                StandardDescriptorSetAllocatorCreateInfo::default(),
            ));
        let pipeline = create_compute_pipeline(
            &device,
            include_bytes!(concat!(env!("OUT_DIR"), "/worldgen.spv")),
        )?;
        let prepare_pipeline = create_compute_pipeline(
            &device,
            include_bytes!(concat!(env!("OUT_DIR"), "/prepare_terrain_y.spv")),
        )?;

        Ok((
            Arc::new(Self {
                device,
                queues,
                queue_family,
                memory_allocator,
                command_allocator,
                descriptor_allocator,
                pipeline,
                prepare_pipeline,
                info: DeviceInfo {
                    name,
                    api_version,
                    vram_bytes,
                },
            }),
            lanes,
            source,
        ))
    }

    pub(super) fn format_vram(&self) -> String {
        if self.info.vram_bytes == 0 {
            String::new()
        } else {
            format!(
                "{:.1} GiB VRAM, ",
                self.info.vram_bytes as f64 / BYTES_PER_GIB
            )
        }
    }

    /// Upload immutable slice data into a device-local storage buffer via a staging copy.
    pub(super) fn upload_storage_slice<T: BufferContents + Copy>(
        self: &Arc<Self>,
        data: &[T],
    ) -> Result<StorageBuffer<T>, String> {
        let staging = Buffer::from_iter(
            Arc::clone(&self.memory_allocator),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.iter().copied(),
        )
        .map_err(|e| format!("failed to create Vulkan staging buffer: {e}"))?;

        let target = Buffer::new_slice::<T>(
            Arc::clone(&self.memory_allocator),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            data.len() as u64,
        )
        .map_err(|e| format!("failed to create Vulkan device buffer: {e}"))?;

        let mut builder = AutoCommandBufferBuilder::primary(
            Arc::clone(&self.command_allocator),
            self.queue_family,
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| format!("failed to begin Vulkan upload command buffer: {e}"))?;

        builder
            .copy_buffer(CopyBufferInfo::buffers(staging, target.clone()))
            .map_err(|e| format!("failed to record Vulkan staging copy: {e}"))?;

        let command_buffer = builder
            .build()
            .map_err(|e| format!("failed to build Vulkan upload command buffer: {e}"))?;

        sync::now(Arc::clone(&self.device))
            .then_execute(Arc::clone(&self.queues[0]), command_buffer)
            .map_err(|e| format!("failed to submit Vulkan upload: {e}"))?
            .then_signal_fence_and_flush()
            .map_err(|e| format!("failed to flush Vulkan upload: {e}"))?
            .wait(None)
            .map_err(|e| format!("failed waiting for Vulkan upload: {e}"))?;

        Ok(target)
    }
}

struct SmearKey {
    plan_id: u64,
    size_y: i32,
    step_y: i32,
    bases: Vec<i32>,
}

struct CachedDescriptor {
    plan_id: u64,
    output_capacity: usize,
    smear_generation: u64,
    set: Arc<DescriptorSet>,
}

/// Per-worker-thread Vulkan compute execution lane with reusable buffers, smear-Y table cache,
/// and cached descriptor sets.
pub(super) struct ComputeLane {
    device: Arc<VulkanDevice>,
    queue: Arc<Queue>,
    params_buffer: Subbuffer<BatchParams>,
    prepare_params_buffer: Subbuffer<PrepareParams>,
    smear_table: Subbuffer<[SmearY]>,
    smear_capacity: usize,
    smear_generation: u64,
    smear_key: Option<SmearKey>,
    output_device: Option<Subbuffer<[f32]>>,
    output_host: Option<Subbuffer<[f32]>>,
    output_capacity: usize,
    cached_descriptor: Option<CachedDescriptor>,
}

impl ComputeLane {
    pub(super) fn new(device: Arc<VulkanDevice>, lane_index: usize) -> Result<Self, String> {
        let queue = Arc::clone(&device.queues[lane_index % device.queues.len()]);
        let params_buffer = Buffer::from_data(
            Arc::clone(&device.memory_allocator),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            BatchParams::default(),
        )
        .map_err(|e| format!("failed to allocate Vulkan batch params buffer: {e}"))?;

        let prepare_params_buffer = Buffer::from_data(
            Arc::clone(&device.memory_allocator),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            PrepareParams::default(),
        )
        .map_err(|e| format!("failed to allocate Vulkan prepare params buffer: {e}"))?;

        let smear_table = Buffer::new_slice::<SmearY>(
            Arc::clone(&device.memory_allocator),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            16,
        )
        .map_err(|e| format!("failed to allocate initial Vulkan smear table: {e}"))?;

        Ok(Self {
            device,
            queue,
            params_buffer,
            prepare_params_buffer,
            smear_table,
            smear_capacity: 16,
            smear_generation: 0,
            smear_key: None,
            output_device: None,
            output_host: None,
            output_capacity: 0,
            cached_descriptor: None,
        })
    }

    fn reserve_output(&mut self, len: usize) -> Result<(), String> {
        if self.output_capacity < len {
            let capacity = len.next_power_of_two().max(WORKGROUP_SIZE);
            let dev = Buffer::new_slice::<f32>(
                Arc::clone(&self.device.memory_allocator),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                capacity as u64,
            )
            .map_err(|e| format!("failed to allocate Vulkan output buffer: {e}"))?;

            let host = Buffer::new_slice::<f32>(
                Arc::clone(&self.device.memory_allocator),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
                },
                capacity as u64,
            )
            .map_err(|e| format!("failed to allocate Vulkan readback buffer: {e}"))?;

            self.output_device = Some(dev);
            self.output_host = Some(host);
            self.output_capacity = capacity;
            self.cached_descriptor = None;
        }
        Ok(())
    }

    fn plan_smear_table(
        &mut self,
        plan_id: u64,
        params: &mut BatchParams,
        batch_len: usize,
    ) -> Result<Option<(PrepareParams, u32, u32)>, String> {
        let total = params.blended.total_count as usize;
        if total == 0 || params.size_y <= 0 {
            params.use_smear_table = 0;
            return Ok(None);
        }
        let size_y = params.size_y as usize;
        let row_entries = total.saturating_mul(size_y);
        if row_entries == 0 || row_entries > MAX_SMEAR_ENTRIES {
            params.use_smear_table = 0;
            return Ok(None);
        }

        if let Some(key) = &self.smear_key
            && key.plan_id == plan_id
            && key.size_y == params.size_y
            && key.step_y == params.step_y
        {
            let mut rows = [0u32; MAX_GRID_CHUNKS];
            let mut all_found = true;
            for (index, row_slot) in rows.iter_mut().enumerate().take(batch_len) {
                let base_y = params.origins[3 * index + 1];
                if let Some(found) = key.bases.iter().position(|&b| b == base_y) {
                    *row_slot = found as u32;
                } else {
                    all_found = false;
                    break;
                }
            }
            if all_found {
                params.table_rows = rows;
                params.use_smear_table = 1;
                return Ok(None);
            }
        }

        let mut bases = Vec::with_capacity(batch_len);
        let mut rows = [0u32; MAX_GRID_CHUNKS];
        for (index, row_slot) in rows.iter_mut().enumerate().take(batch_len) {
            let base_y = params.origins[3 * index + 1];
            let row = bases.iter().position(|&b| b == base_y).unwrap_or_else(|| {
                bases.push(base_y);
                bases.len() - 1
            });
            *row_slot = row as u32;
        }
        let needed = row_entries.saturating_mul(bases.len());
        if needed > MAX_SMEAR_ENTRIES {
            params.use_smear_table = 0;
            return Ok(None);
        }

        if self.smear_capacity < needed {
            let capacity = needed.next_power_of_two().max(16);
            self.smear_table = Buffer::new_slice::<SmearY>(
                Arc::clone(&self.device.memory_allocator),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                capacity as u64,
            )
            .map_err(|e| format!("failed to allocate Vulkan smear table: {e}"))?;
            self.smear_capacity = capacity;
            self.smear_generation += 1;
            self.cached_descriptor = None;
        }

        let mut base_array = [0i32; MAX_GRID_CHUNKS];
        for (dst, &src) in base_array.iter_mut().zip(&bases) {
            *dst = src;
        }
        let prep = PrepareParams {
            layer_offset: params.blended.layer_offset,
            smear_offset: params.blended.smear_offset,
            size_y: params.size_y,
            step_y: params.step_y,
            row_entries: row_entries as u32,
            bases: base_array,
        };
        let row_count = bases.len() as u32;
        self.smear_key = Some(SmearKey {
            plan_id,
            size_y: params.size_y,
            step_y: params.step_y,
            bases,
        });
        params.table_rows = rows;
        params.use_smear_table = 1;
        Ok(Some((
            prep,
            (row_entries as u32).div_ceil(PREPARE_WORKGROUP_SIZE),
            row_count,
        )))
    }

    #[expect(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(super) fn execute<L: BufferContents>(
        &mut self,
        plan_id: u64,
        layers: &StorageBuffer<L>,
        smears: &StorageBuffer<f64>,
        params: &mut BatchParams,
        batch_len: usize,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        output_len: usize,
    ) -> Result<Vec<f32>, String> {
        self.reserve_output(output_len)?;
        let prepare_job = self.plan_smear_table(plan_id, params, batch_len)?;

        {
            let mut guard = self
                .params_buffer
                .write()
                .map_err(|e| format!("failed to map Vulkan params buffer: {e}"))?;
            *guard = *params;
        };

        if let Some((prep_params, _, _)) = prepare_job {
            let mut guard = self
                .prepare_params_buffer
                .write()
                .map_err(|e| format!("failed to map Vulkan prepare params buffer: {e}"))?;
            *guard = prep_params;
        }

        let Some(output_device) = self.output_device.as_ref() else {
            return Ok(Vec::new());
        };
        let Some(output_host) = self.output_host.as_ref() else {
            return Ok(Vec::new());
        };

        let descriptor_set = match &self.cached_descriptor {
            Some(cached)
                if cached.plan_id == plan_id
                    && cached.output_capacity == self.output_capacity
                    && cached.smear_generation == self.smear_generation =>
            {
                Arc::clone(&cached.set)
            }
            _ => {
                let layout = self
                    .device
                    .pipeline
                    .layout()
                    .set_layouts()
                    .first()
                    .ok_or_else(|| String::from("Vulkan pipeline missing descriptor set layout"))?;
                let set = DescriptorSet::new(
                    Arc::clone(&self.device.descriptor_allocator),
                    Arc::clone(layout),
                    [
                        WriteDescriptorSet::buffer(0, layers.clone()),
                        WriteDescriptorSet::buffer(1, smears.clone()),
                        WriteDescriptorSet::buffer(2, self.params_buffer.clone()),
                        WriteDescriptorSet::buffer(3, output_device.clone()),
                        WriteDescriptorSet::buffer(4, self.smear_table.clone()),
                    ],
                    [],
                )
                .map_err(|e| format!("failed to create Vulkan descriptor set: {e}"))?;
                self.cached_descriptor = Some(CachedDescriptor {
                    plan_id,
                    output_capacity: self.output_capacity,
                    smear_generation: self.smear_generation,
                    set: Arc::clone(&set),
                });
                set
            }
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            Arc::clone(&self.device.command_allocator),
            self.device.queue_family,
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| format!("failed to begin Vulkan compute command buffer: {e}"))?;

        if let Some((_, prep_groups_x, prep_groups_y)) = prepare_job {
            let prep_layout = self
                .device
                .prepare_pipeline
                .layout()
                .set_layouts()
                .first()
                .ok_or_else(|| {
                    String::from("Vulkan prepare pipeline missing descriptor set layout")
                })?;
            let prep_set = DescriptorSet::new(
                Arc::clone(&self.device.descriptor_allocator),
                Arc::clone(prep_layout),
                [
                    WriteDescriptorSet::buffer(0, layers.clone()),
                    WriteDescriptorSet::buffer(1, smears.clone()),
                    WriteDescriptorSet::buffer(2, self.prepare_params_buffer.clone()),
                    WriteDescriptorSet::buffer(3, self.smear_table.clone()),
                ],
                [],
            )
            .map_err(|e| format!("failed to create Vulkan prepare descriptor set: {e}"))?;

            builder
                .bind_pipeline_compute(Arc::clone(&self.device.prepare_pipeline))
                .map_err(|e| format!("failed to bind Vulkan prepare pipeline: {e}"))?
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    Arc::clone(self.device.prepare_pipeline.layout()),
                    0,
                    prep_set,
                )
                .map_err(|e| format!("failed to bind Vulkan prepare descriptor set: {e}"))?;

            // SAFETY: `prepare_terrain_y.comp` writes `smear_table[..row_entries * row_count]`
            // within the allocated `smear_table` buffer.
            unsafe {
                builder
                    .dispatch([prep_groups_x, prep_groups_y, 1])
                    .map_err(|e| format!("failed to record Vulkan prepare dispatch: {e}"))?;
            };
        }

        builder
            .bind_pipeline_compute(Arc::clone(&self.device.pipeline))
            .map_err(|e| format!("failed to bind Vulkan pipeline: {e}"))?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                Arc::clone(self.device.pipeline.layout()),
                0,
                descriptor_set,
            )
            .map_err(|e| format!("failed to bind Vulkan descriptor set: {e}"))?;

        // SAFETY: `worldgen.comp` reads within `layers`, `smears`, `params`, and `smear_table`,
        // and writes within `output_device[..output_len]` for `[grid_x, grid_y, grid_z]`.
        unsafe {
            builder
                .dispatch([grid_x, grid_y, grid_z])
                .map_err(|e| format!("failed to record Vulkan dispatch: {e}"))?;
        };

        let output_slice_len = output_len as u64;
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                output_device.clone().slice(..output_slice_len),
                output_host.clone().slice(..output_slice_len),
            ))
            .map_err(|e| format!("failed to record Vulkan readback copy: {e}"))?;

        let command_buffer = builder
            .build()
            .map_err(|e| format!("failed to build Vulkan compute command buffer: {e}"))?;

        sync::now(Arc::clone(&self.device.device))
            .then_execute(Arc::clone(&self.queue), command_buffer)
            .map_err(|e| format!("failed to submit Vulkan compute batch: {e}"))?
            .then_signal_fence_and_flush()
            .map_err(|e| format!("failed to flush Vulkan compute batch: {e}"))?
            .wait(None)
            .map_err(|e| format!("failed waiting for Vulkan compute batch: {e}"))?;

        let read_guard = output_host
            .read()
            .map_err(|e| format!("failed to read Vulkan output buffer: {e}"))?;
        Ok(read_guard[..output_len].to_vec())
    }
}

fn create_compute_pipeline(
    device: &Arc<Device>,
    spv_bytes: &[u8],
) -> Result<Arc<ComputePipeline>, String> {
    if !spv_bytes.len().is_multiple_of(4) {
        return Err(String::from(
            "embedded Vulkan SPIR-V size is not 4-byte aligned",
        ));
    }
    let (chunks, _) = spv_bytes.as_chunks::<4>();
    let words: Vec<u32> = chunks.iter().copied().map(u32::from_le_bytes).collect();

    // SAFETY: SPIR-V bytes are compiled at build time from GLSL compute shaders.
    let shader =
        unsafe { ShaderModule::new(Arc::clone(device), ShaderModuleCreateInfo::new(&words)) }
            .map_err(|e| format!("failed to load Vulkan shader module: {e}"))?;
    let entry = shader
        .entry_point("main")
        .ok_or_else(|| String::from("Vulkan shader missing main entry point"))?;
    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout = PipelineLayout::new(
        Arc::clone(device),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(Arc::clone(device))
            .map_err(|e| format!("failed to build Vulkan pipeline layout info: {e}"))?,
    )
    .map_err(|e| format!("failed to create Vulkan pipeline layout: {e}"))?;

    ComputePipeline::new(
        Arc::clone(device),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .map_err(|e| format!("failed to create Vulkan compute pipeline: {e}"))
}

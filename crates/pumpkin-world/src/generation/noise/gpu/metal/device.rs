//! Apple Metal compute device and unified-memory worker lane via `objc2-metal`.

use super::abi::{BatchParams, CoordX, CoordY, CoordZ, MetalLayer};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, ns_string};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCopyAllDevices, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use std::ptr::NonNull;
use std::sync::Arc;

const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const THREADGROUP_WIDTH: usize = 64;
const MAX_LANES: usize = 8;

const SHADER_SOURCE: &str = include_str!("worldgen.metal");

pub(super) struct DeviceInfo {
    pub name: String,
    pub memory_bytes: u64,
    pub unified_memory: bool,
}

pub(super) struct MetalBuffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    capacity_bytes: usize,
}

// SAFETY: Metal buffers are either immutable after upload or owned exclusively by one ComputeLane.
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl MetalBuffer {
    fn write_slice<T: Copy>(&self, data: &[T]) {
        let byte_len = std::mem::size_of_val(data);
        debug_assert!(byte_len <= self.capacity_bytes);
        if byte_len == 0 {
            return;
        }
        let dst = self.raw.contents().as_ptr().cast::<u8>();
        // SAFETY: `raw` is a `StorageModeShared` buffer with at least `byte_len` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), dst, byte_len);
        }
    }

    fn write_value<T: Copy>(&self, value: &T) {
        self.write_slice(std::slice::from_ref(value));
    }

    fn read_f32(&self, count: usize) -> Vec<f32> {
        let byte_len = count * std::mem::size_of::<f32>();
        debug_assert!(byte_len <= self.capacity_bytes);
        let mut out = vec![0.0f32; count];
        if byte_len == 0 {
            return out;
        }
        let src = self.raw.contents().as_ptr().cast::<u8>();
        // SAFETY: Command buffer completed before read; `src` holds `count` initialized f32s.
        unsafe {
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr().cast::<u8>(), byte_len);
        }
        out
    }
}

pub(super) struct MetalDevice {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub info: DeviceInfo,
}

// SAFETY: `MTLDevice` and `MTLComputePipelineState` are thread-safe in Apple's Metal API.
unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

impl MetalDevice {
    pub(super) fn create(
        device_index: usize,
        configured_streams: usize,
    ) -> Result<(Arc<Self>, usize, &'static str), String> {
        let devices = MTLCopyAllDevices();
        let count = devices.len();
        if count == 0 {
            return Err(String::from("no Metal-capable GPU detected"));
        }
        if device_index >= count {
            return Err(if count == 1 {
                format!("metal:{device_index} not found (only device 0 is available)")
            } else {
                format!("metal:{device_index} not found (available devices: 0..{count})")
            });
        }

        let device = devices.objectAtIndex(device_index);
        let name = device.name().to_string();
        let memory_bytes = device.recommendedMaxWorkingSetSize();
        let unified_memory = device.hasUnifiedMemory();

        let source = NSString::from_str(SHADER_SOURCE);
        let options = MTLCompileOptions::new();
        // Preserve strict IEEE-754 f32 rounding and disable FMA contraction for vanilla parity.
        #[allow(deprecated)]
        options.setFastMathEnabled(false);

        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|e| format!("failed to compile Metal shader on metal:{device_index}: {e}"))?;
        let function = library
            .newFunctionWithName(ns_string!("fill_worldgen_metal"))
            .ok_or_else(|| String::from("Metal shader missing fill_worldgen_metal kernel"))?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                format!("failed to create Metal compute pipeline on metal:{device_index}: {e}")
            })?;

        let host_cpus = std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get);
        let (lanes, streams_source) = if configured_streams > 0 {
            (configured_streams.min(MAX_LANES), "config")
        } else {
            let auto = 2usize.min((host_cpus / 2).max(1));
            (auto, "auto")
        };

        Ok((
            Arc::new(Self {
                device,
                pipeline,
                info: DeviceInfo {
                    name,
                    memory_bytes,
                    unified_memory,
                },
            }),
            lanes,
            streams_source,
        ))
    }

    pub(super) fn format_memory(&self) -> String {
        if self.info.memory_bytes == 0 {
            String::new()
        } else {
            let kind = if self.info.unified_memory {
                "unified memory"
            } else {
                "VRAM"
            };
            format!(
                "{:.1} GiB {kind}, ",
                self.info.memory_bytes as f64 / BYTES_PER_GIB
            )
        }
    }

    fn alloc_shared(&self, bytes: usize) -> Result<MetalBuffer, String> {
        let len = bytes.max(16);
        let raw = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| format!("failed to allocate {len}-byte Metal shared buffer"))?;
        Ok(MetalBuffer {
            raw,
            capacity_bytes: len,
        })
    }

    pub(super) fn upload_layers(&self, layers: &[MetalLayer]) -> Result<MetalBuffer, String> {
        let byte_len = std::mem::size_of_val(layers).max(16);
        let ptr = NonNull::new(layers.as_ptr().cast_mut().cast())
            .ok_or_else(|| String::from("empty Metal layer slice"))?;
        // SAFETY: `ptr` points to `byte_len` initialized bytes for the duration of the copy.
        let raw = unsafe {
            self.device.newBufferWithBytes_length_options(
                ptr,
                byte_len,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| format!("failed to upload {byte_len}-byte Metal layer buffer"))?;
        Ok(MetalBuffer {
            raw,
            capacity_bytes: byte_len,
        })
    }
}

/// Per-worker-thread Metal compute lane with reusable shared-memory buffers.
pub(super) struct ComputeLane {
    device: Arc<MetalDevice>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    params_buf: MetalBuffer,
    coord_x_buf: Option<MetalBuffer>,
    coord_z_buf: Option<MetalBuffer>,
    coord_y_buf: Option<MetalBuffer>,
    output_buf: Option<MetalBuffer>,
}

// SAFETY: Each ComputeLane is owned exclusively by a single worker thread.
unsafe impl Send for ComputeLane {}

impl ComputeLane {
    pub(super) fn new(device: Arc<MetalDevice>) -> Result<Self, String> {
        let queue = device
            .device
            .newCommandQueue()
            .ok_or_else(|| String::from("failed to create Metal command queue"))?;
        let params_buf = device.alloc_shared(std::mem::size_of::<BatchParams>())?;
        Ok(Self {
            device,
            queue,
            params_buf,
            coord_x_buf: None,
            coord_z_buf: None,
            coord_y_buf: None,
            output_buf: None,
        })
    }

    fn ensure_buf<'a>(
        device: &MetalDevice,
        slot: &'a mut Option<MetalBuffer>,
        needed_bytes: usize,
    ) -> Result<&'a MetalBuffer, String> {
        let needs_alloc = slot
            .as_ref()
            .is_none_or(|buf| buf.capacity_bytes < needed_bytes);
        if needs_alloc {
            let alloc = needed_bytes.next_power_of_two().max(64);
            *slot = Some(device.alloc_shared(alloc)?);
        }
        slot.as_ref()
            .ok_or_else(|| String::from("Metal buffer allocation missing"))
    }

    #[expect(clippy::too_many_arguments)]
    pub(super) fn execute(
        &mut self,
        layers: &MetalBuffer,
        coords_x: &[CoordX],
        coords_z: &[CoordZ],
        coords_y: &[CoordY],
        params: &BatchParams,
        grid_x: usize,
        grid_y: usize,
        grid_z: usize,
        output_len: usize,
    ) -> Result<Vec<f32>, String> {
        self.params_buf.write_value(params);

        let cx_buf = Self::ensure_buf(
            &self.device,
            &mut self.coord_x_buf,
            std::mem::size_of_val(coords_x),
        )?;
        cx_buf.write_slice(coords_x);

        let cz_buf = Self::ensure_buf(
            &self.device,
            &mut self.coord_z_buf,
            std::mem::size_of_val(coords_z),
        )?;
        cz_buf.write_slice(coords_z);

        let cy_buf = Self::ensure_buf(
            &self.device,
            &mut self.coord_y_buf,
            std::mem::size_of_val(coords_y),
        )?;
        cy_buf.write_slice(coords_y);

        let out_buf = Self::ensure_buf(
            &self.device,
            &mut self.output_buf,
            output_len * std::mem::size_of::<f32>(),
        )?;

        let command_buffer = self
            .queue
            .commandBuffer()
            .ok_or_else(|| String::from("failed to create Metal command buffer"))?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| String::from("failed to create Metal compute encoder"))?;

        encoder.setComputePipelineState(&self.device.pipeline);
        // SAFETY: All buffers remain alive until `waitUntilCompleted` finishes below.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&layers.raw), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&cx_buf.raw), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&cz_buf.raw), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&cy_buf.raw), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&self.params_buf.raw), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&out_buf.raw), 0, 5);
        }

        let threadgroups = MTLSize {
            width: grid_x,
            height: grid_y,
            depth: grid_z,
        };
        let threads_per_group = MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_group);
        encoder.endEncoding();

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        if let Some(error) = command_buffer.error() {
            return Err(format!("Metal command buffer failed: {error}"));
        }

        Ok(out_buf.read_f32(output_len))
    }
}

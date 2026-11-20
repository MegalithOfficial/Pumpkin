//! GPU-accelerated terrain generation backends.
//!
//! Each backend lives in its own submodule (`cuda`, and future `vulkan` / `metal`)
//! behind its Cargo feature gate, while this module owns backend dispatch, shared
//! resident caches, and startup initialization.

#[cfg(feature = "cuda-worldgen")]
pub(crate) mod cuda;

use pumpkin_config::world::{WorldgenBackend, WorldgenConfig};
#[cfg(feature = "gpu-worldgen")]
use {
    super::router::{
        chunk_noise_router::ChunkNoiseFunctionComponent, proto_noise_router::ProtoNoiseRouter,
    },
    crate::generation::generator::VanillaGenerator,
    pumpkin_util::noise::volume::DensityVolume,
    std::sync::{Arc, OnceLock, atomic::AtomicBool},
};

/// Initialize the process-wide world generation backend from server config.
pub(in crate::generation) fn initialize(config: WorldgenConfig) {
    match config.backend {
        WorldgenBackend::Cpu => tracing::info!("World generation uses CPU"),
        WorldgenBackend::Cuda => {
            #[cfg(feature = "cuda-worldgen")]
            cuda::initialize(config);

            #[cfg(not(feature = "cuda-worldgen"))]
            tracing::warn!(
                "World generation falling back to CPU: this build lacks the cuda-worldgen feature"
            );
        }
        WorldgenBackend::Vulkan => {
            tracing::warn!(
                "World generation falling back to CPU: Vulkan world generation is not implemented yet"
            );
        }
        WorldgenBackend::Metal => {
            tracing::warn!(
                "World generation falling back to CPU: Metal world generation is not implemented yet"
            );
        }
    }
}

/// Per-sampler resident GPU noise layer cache across enabled backends.
#[cfg(feature = "gpu-worldgen")]
pub(crate) struct ResidentLayers {
    #[cfg(feature = "cuda-worldgen")]
    pub(crate) cuda: cuda::ResidentLayers,
}

#[cfg(feature = "gpu-worldgen")]
impl ResidentLayers {
    pub(crate) const fn new() -> Self {
        Self {
            #[cfg(feature = "cuda-worldgen")]
            cuda: OnceLock::new(),
        }
    }
}

/// Per-sampler resident GPU blended-noise cache across enabled backends.
#[cfg(feature = "gpu-worldgen")]
pub(crate) struct BlendedResident {
    #[cfg(feature = "cuda-worldgen")]
    pub(crate) cuda: cuda::BlendedResident,
}

#[cfg(feature = "gpu-worldgen")]
impl BlendedResident {
    pub(crate) const fn new() -> Self {
        Self {
            #[cfg(feature = "cuda-worldgen")]
            cuda: OnceLock::new(),
        }
    }
}

/// Per-generator cached GPU grid plan across enabled backends.
#[cfg(feature = "gpu-worldgen")]
pub(crate) struct PlanCache {
    #[cfg(feature = "cuda-worldgen")]
    pub(crate) cuda: OnceLock<Option<Arc<cuda::grid::Plan>>>,
}

#[cfg(feature = "gpu-worldgen")]
impl PlanCache {
    pub(crate) const fn new() -> Self {
        Self {
            #[cfg(feature = "cuda-worldgen")]
            cuda: OnceLock::new(),
        }
    }
}

/// A chunk grid pre-submitted to the active GPU backend by the lookahead thread.
#[cfg(feature = "gpu-worldgen")]
pub(crate) enum PreparedGrid {
    #[cfg(feature = "cuda-worldgen")]
    Cuda(cuda::grid::PreparedGrid),
}

#[cfg(feature = "gpu-worldgen")]
impl PreparedGrid {
    pub(crate) fn activate(self, generator: &VanillaGenerator) -> Option<EarlyGridScope<'_>> {
        match self {
            #[cfg(feature = "cuda-worldgen")]
            Self::Cuda(grid) => grid.activate(generator).map(EarlyGridScope::Cuda),
        }
    }
}

/// Thread-local guard keeping an early-prefetched GPU grid active during a chunk's Noise stage.
#[cfg(feature = "gpu-worldgen")]
pub(crate) enum EarlyGridScope<'a> {
    #[cfg(feature = "cuda-worldgen")]
    Cuda(cuda::grid::EarlyGridScope<'a>),
}

#[cfg(feature = "gpu-worldgen")]
pub(crate) fn lookahead_enabled() -> bool {
    #[cfg(feature = "cuda-worldgen")]
    if cuda::scheduled_lookahead_enabled() {
        return true;
    }
    false
}

#[cfg(feature = "gpu-worldgen")]
pub(crate) fn prepare_chunk(
    generator: &VanillaGenerator,
    chunk_x: i32,
    chunk_z: i32,
    cancel: Option<Arc<AtomicBool>>,
) -> Option<PreparedGrid> {
    #[cfg(feature = "cuda-worldgen")]
    if let Some(grid) = cuda::grid::prepare_chunk(generator, chunk_x, chunk_z, cancel) {
        return Some(PreparedGrid::Cuda(grid));
    }
    None
}

#[cfg(feature = "gpu-worldgen")]
pub(crate) fn prefetch_chunk(
    generator: &VanillaGenerator,
    chunk_x: i32,
    chunk_z: i32,
) -> Option<EarlyGridScope<'_>> {
    #[cfg(feature = "cuda-worldgen")]
    if let Some(scope) = cuda::grid::prefetch_chunk(generator, chunk_x, chunk_z) {
        return Some(EarlyGridScope::Cuda(scope));
    }
    None
}

#[cfg(feature = "gpu-worldgen")]
pub(crate) fn try_volume(
    stack: &mut [ChunkNoiseFunctionComponent<'_>],
    out: &mut [f32],
    volume: &DensityVolume,
) -> bool {
    #[cfg(feature = "cuda-worldgen")]
    if cuda::try_volume(stack, out, volume) {
        return true;
    }
    false
}

#[cfg(feature = "gpu-worldgen")]
pub(crate) fn try_sample_density(
    stack: &mut [ChunkNoiseFunctionComponent<'_>],
    out: &mut [f32],
    volume: &DensityVolume,
    source: &ProtoNoiseRouter,
) -> bool {
    #[cfg(feature = "cuda-worldgen")]
    if cuda::grid::try_sample_density(stack, out, volume, source) {
        return true;
    }
    false
}

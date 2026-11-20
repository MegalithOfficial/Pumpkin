//! GPU-accelerated terrain generation backends.
//!
//! Each backend lives in its own submodule (`cuda`, and future `vulkan` / `metal`)
//! behind its Cargo feature gate, while this module owns backend dispatch and
//! startup initialization.

#[cfg(feature = "cuda-worldgen")]
pub(crate) mod cuda;

use pumpkin_config::world::{WorldgenBackend, WorldgenConfig};

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

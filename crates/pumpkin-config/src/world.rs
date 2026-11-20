use serde::{Deserialize, Deserializer, Serialize};

use crate::{chunk::ChunkConfig, lighting::LightingEngineConfig};

/// Configuration for world and level-specific settings.
///
/// Currently, it includes chunk-related options; more settings may be added later.
#[derive(Deserialize, Serialize, Clone)]
pub struct LevelConfig {
    /// Configuration for chunk behaviour and management.
    pub chunk: ChunkConfig,
    /// Process-wide settings for terrain generation.
    #[serde(default)]
    pub generation: WorldgenConfig,
    /// Configuration for lighting engine propagation mode.
    #[serde(default)]
    pub lighting: LightingEngineConfig,
    /// Number of ticks between autosave checks. If 0, autosave is disabled.
    #[serde(default = "default_autosave_ticks")]
    pub autosave_ticks: u64,
    /// Margins for ticking entities in parallel islands.
    #[serde(default)]
    pub entity_islands: EntityIslandsConfig,
    // TODO: More options
}

/// Settings for terrain generation backend selection.
#[derive(Deserialize, Serialize, Debug, Clone, Copy, Default, PartialEq, Eq)]
#[serde(default)]
pub struct WorldgenConfig {
    /// Backend used to generate terrain.
    pub backend: WorldgenBackend,
    /// GPU device index (`0` = first GPU, `1` = second GPU, ...). Ignored on CPU.
    pub device: usize,
    /// Number of concurrent GPU worker streams (`0` = auto based on GPU SM count). Ignored on CPU.
    pub streams: usize,
}

/// Backend used to generate terrain.
#[derive(Deserialize, Serialize, Debug, Clone, Copy, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum WorldgenBackend {
    /// Run terrain generation on the CPU.
    #[default]
    Cpu,
    /// Offload supported density functions to NVIDIA CUDA.
    Cuda,
    /// Offload supported density functions to Vulkan compute.
    Vulkan,
    /// Offload supported density functions to Apple Metal.
    Metal,
}

/// Safety margins for ticking entities in parallel islands. Lower values tick more entities in
/// parallel but can let interactions run out of tick order.
#[derive(Deserialize, Serialize, Clone, Copy)]
#[serde(default)]
pub struct EntityIslandsConfig {
    /// Blocks an entity whose tick is not audited can reach beyond its follow range.
    #[serde(deserialize_with = "EntityIslandsConfig::deserialize_unaudited_reach")]
    pub unaudited_reach: f64,
    /// Blocks a hit from such an entity can throw an audited one in a tick. Raise it for TNT
    /// cannons.
    #[serde(deserialize_with = "EntityIslandsConfig::deserialize_knockback_reach")]
    pub knockback_reach: f64,
}

impl EntityIslandsConfig {
    fn deserialize_unaudited_reach<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<f64, D::Error> {
        deserialize_margin(deserializer, Self::default().unaudited_reach)
    }

    fn deserialize_knockback_reach<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<f64, D::Error> {
        deserialize_margin(deserializer, Self::default().knockback_reach)
    }
}

fn deserialize_margin<'de, D: Deserializer<'de>>(
    deserializer: D,
    fallback: f64,
) -> Result<f64, D::Error> {
    let value = f64::deserialize(deserializer)?;
    Ok(if value.is_finite() {
        value.max(0.0)
    } else {
        fallback
    })
}

impl Default for EntityIslandsConfig {
    fn default() -> Self {
        Self {
            unaudited_reach: 16.0,
            knockback_reach: 4.0,
        }
    }
}

const fn default_autosave_ticks() -> u64 {
    6000 // Default to 5 minutes at 20 TPS
}

impl Default for LevelConfig {
    fn default() -> Self {
        Self {
            chunk: ChunkConfig::default(),
            generation: WorldgenConfig::default(),
            lighting: LightingEngineConfig::default(),
            autosave_ticks: default_autosave_ticks(),
            entity_islands: EntityIslandsConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worldgen_config_defaults_to_cpu_and_accepts_gpu_backends() {
        let mut config = toml::Value::try_from(LevelConfig::default()).unwrap();
        config.as_table_mut().unwrap().remove("generation");
        let legacy: LevelConfig = config.clone().try_into().unwrap();
        assert_eq!(legacy.generation.backend, WorldgenBackend::Cpu);
        assert_eq!(legacy.generation.device, 0);
        assert_eq!(legacy.generation.streams, 0);

        for (name, expected, device, streams) in [
            ("cuda", WorldgenBackend::Cuda, 1, 4),
            ("vulkan", WorldgenBackend::Vulkan, 2, 0),
            ("metal", WorldgenBackend::Metal, 0, 2),
        ] {
            let mut generation = toml::map::Map::new();
            generation.insert("backend".to_owned(), toml::Value::String(name.to_owned()));
            generation.insert("device".to_owned(), toml::Value::Integer(device));
            generation.insert("streams".to_owned(), toml::Value::Integer(streams));
            let mut next = config.clone();
            next.as_table_mut()
                .unwrap()
                .insert("generation".to_owned(), toml::Value::Table(generation));
            let parsed: LevelConfig = next.try_into().unwrap();
            assert_eq!(parsed.generation.backend, expected);
            assert_eq!(parsed.generation.device, usize::try_from(device).unwrap());
            assert_eq!(parsed.generation.streams, usize::try_from(streams).unwrap());
        }
    }

    #[test]
    fn default_level_config_enables_autosave() {
        assert_eq!(LevelConfig::default().autosave_ticks, 6000);
    }

    #[test]
    fn generated_config_round_trip_keeps_autosave_enabled() {
        let generated = toml::to_string(&LevelConfig::default()).unwrap();
        let reloaded: LevelConfig = toml::from_str(&generated).unwrap();
        assert_eq!(reloaded.autosave_ticks, 6000);
    }

    #[test]
    fn an_explicit_zero_still_disables_autosave() {
        let disabled = LevelConfig {
            autosave_ticks: 0,
            ..LevelConfig::default()
        };
        let generated = toml::to_string(&disabled).unwrap();
        let reloaded: LevelConfig = toml::from_str(&generated).unwrap();
        assert_eq!(reloaded.autosave_ticks, 0);
    }
}

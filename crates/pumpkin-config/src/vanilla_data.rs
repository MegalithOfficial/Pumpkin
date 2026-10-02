use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Where the vanilla data prepared from Mojang's server jar is kept.
#[derive(Deserialize, Serialize)]
#[serde(default)]
pub struct VanillaDataConfig {
    /// Path to `vanilla.pak`, relative to the server directory unless absolute.
    pub pack_path: PathBuf,
}

impl Default for VanillaDataConfig {
    fn default() -> Self {
        Self {
            pack_path: PathBuf::from("vanilla.pak"),
        }
    }
}

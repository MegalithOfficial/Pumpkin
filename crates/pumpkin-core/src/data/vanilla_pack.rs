//! Makes sure `vanilla.pak` exists at startup, building it from Mojang's server jar if needed.

use std::path::Path;
use std::time::Duration;

use pumpkin_config::vanilla_data::VanillaDataConfig;
use pumpkin_data::packet::CURRENT_MC_VERSION;
use pumpkin_world::chunk::format::anvil::WORLD_DATA_VERSION;
use pumpkin_world::vanilla_pack::{self, PackError, VanillaPack, module};
use serde::Deserialize;
use sha1::{Digest, Sha1};
use tracing::{error, info};

const VERSION_MANIFEST_URL: &str =
    "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json";
const MAX_SERVER_JAR_LEN: u64 = 512 * 1024 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, thiserror::Error)]
pub enum PrepareError {
    #[error("download failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Minecraft {0} is not in Mojang's version manifest")]
    UnknownVersion(String),
    #[error("server jar is {0} bytes, which is more than allowed")]
    TooLarge(u64),
    #[error("downloaded server jar does not match Mojang's checksum")]
    ChecksumMismatch,
    #[error("prepared pack is still not usable: {0}")]
    StillUnusable(PackError),
    #[error(transparent)]
    Pack(#[from] PackError),
    #[error("preparation task failed: {0}")]
    Join(#[from] tokio::task::JoinError),
}

#[derive(Deserialize)]
struct VersionManifest {
    versions: Vec<ManifestVersion>,
}

#[derive(Deserialize)]
struct ManifestVersion {
    id: String,
    url: String,
}

#[derive(Deserialize)]
struct VersionJson {
    downloads: VersionDownloads,
}

#[derive(Deserialize)]
struct VersionDownloads {
    server: Download,
}

#[derive(Deserialize)]
struct Download {
    sha1: String,
    size: u64,
    url: String,
}

/// Opens the configured `vanilla.pak`, preparing it first if it's missing or outdated.
pub async fn load_or_prepare(
    server_dir: &Path,
    config: &VanillaDataConfig,
) -> Result<(), PrepareError> {
    let path = server_dir.join(&config.pack_path);
    let pack = match open(&path) {
        Ok(pack) => pack,
        Err(reason) => {
            match reason {
                PackError::Io(e) if e.kind() == std::io::ErrorKind::NotFound => info!(
                    "No vanilla data at {}, preparing it for Minecraft {CURRENT_MC_VERSION}",
                    path.display()
                ),
                reason => {
                    info!("Preparing vanilla data for Minecraft {CURRENT_MC_VERSION}: {reason}");
                }
            }
            prepare(&path).await?;
            open(&path).map_err(PrepareError::StillUnusable)?
        }
    };

    vanilla_pack::install(pack);
    // load the index off the startup path, it's usually ready before anything needs it
    let spawned = std::thread::Builder::new()
        .name("vanilla-pack-index".into())
        .spawn(|| {
            if let Some(pack) = vanilla_pack::installed()
                && let Err(e) = pack.load_index()
            {
                error!("Failed to load the vanilla pack index: {e}");
            }
        });
    if let Err(e) = spawned {
        error!("Failed to start loading the vanilla pack index early: {e}");
    }
    Ok(())
}

fn open(path: &Path) -> Result<VanillaPack, PackError> {
    let pack = VanillaPack::open(path, WORLD_DATA_VERSION)?;
    if pack.modules() & module::PREPARED != module::PREPARED {
        return Err(PackError::MissingModules {
            missing: module::PREPARED & !pack.modules(),
        });
    }
    Ok(pack)
}

async fn prepare(path: &Path) -> Result<(), PrepareError> {
    let client = pumpkin_auth::client_builder()
        .user_agent("Pumpkin-MC")
        .connect_timeout(CONNECT_TIMEOUT)
        .build()?;

    let version_id = CURRENT_MC_VERSION.to_string();
    let manifest: VersionManifest = client
        .get(VERSION_MANIFEST_URL)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let version = manifest
        .versions
        .into_iter()
        .find(|v| v.id == version_id)
        .ok_or(PrepareError::UnknownVersion(version_id))?;
    let server = client
        .get(&version.url)
        .send()
        .await?
        .error_for_status()?
        .json::<VersionJson>()
        .await?
        .downloads
        .server;
    if server.size > MAX_SERVER_JAR_LEN {
        return Err(PrepareError::TooLarge(server.size));
    }

    let jar = download(&client, &server).await?;
    let sha1: [u8; 20] = Sha1::digest(&jar).into();
    if hex::encode(sha1) != server.sha1.to_ascii_lowercase() {
        return Err(PrepareError::ChecksumMismatch);
    }

    info!("Converting vanilla data");
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(PackError::from)?;
    }
    let path = path.to_owned();
    let count = tokio::task::spawn_blocking(move || {
        vanilla_pack::build_from_server_jar(&jar, sha1, WORLD_DATA_VERSION, &path)
    })
    .await??;
    info!("Prepared vanilla data ({count} resources)");
    Ok(())
}

async fn download(client: &reqwest::Client, server: &Download) -> Result<Vec<u8>, PrepareError> {
    info!(
        "Downloading the Minecraft {CURRENT_MC_VERSION} server jar ({} MB) from Mojang",
        server.size / (1024 * 1024)
    );
    let mut response = client.get(&server.url).send().await?.error_for_status()?;
    let mut jar = Vec::with_capacity(server.size as usize);
    let mut next_report = 10;
    while let Some(chunk) = response.chunk().await? {
        if (jar.len() + chunk.len()) as u64 > server.size {
            return Err(PrepareError::TooLarge(server.size));
        }
        jar.extend_from_slice(&chunk);
        let percent = jar.len() as u64 * 100 / server.size.max(1);
        if percent >= next_report {
            info!("Downloading server jar: {percent}%");
            next_report = percent / 10 * 10 + 10;
        }
    }
    Ok(jar)
}

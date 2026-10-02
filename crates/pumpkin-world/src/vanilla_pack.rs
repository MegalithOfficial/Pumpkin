//! Vanilla resources prepared from the official server jar into `vanilla.pak`.
//!
//! Layout, little endian:
//!
//! ```text
//! 0   [u8; 8]  magic
//! 8   u16      format version
//! 10  u16      reserved
//! 12  i32      Minecraft data version
//! 16  u32      entry count
//! 20  u32      index length in bytes
//! 24  u64      xxh64 of the index
//! 32  [u8; 20] sha1 of the source jar
//! 52  u64      bitmask of the modules the pack contains
//! 60  [u8; 4]  reserved
//! 64  index    record table, then key table
//! ..  blobs
//! ```
//!
//! Records are 32 bytes, sorted by (kind, key): kind u8, encoding u8, key length u16, key offset
//! u32, blob offset u64, blob length u32, reserved u32, blob xxh64 u64. Fixed size records mean
//! loading the index is one read and one hash, no per entry parsing.

use std::cmp::Ordering;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::OnceLock;

use sha2::{Digest, Sha256};
use xxhash_rust::xxh64::xxh64;

const MAGIC: [u8; 8] = *b"PKVANPAK";
pub const FORMAT_VERSION: u16 = 1;
const HEADER_LEN: usize = 64;
const RECORD_LEN: usize = 32;
// keep a corrupt header from making us allocate gigabytes
const MAX_INDEX_LEN: usize = 64 * 1024 * 1024;
const MAX_ENTRY_LEN: u32 = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ResourceKind {
    Structure = 1,
    LootTable = 2,
}

/// Groups of content a pack holds, recorded as a bitmask in the header.
///
/// When a newer Pumpkin reads a group an older pack doesn't have, startup prepares the pack again.
/// Bit positions are part of the format, so new groups only ever get new bits.
pub mod module {
    pub const STRUCTURES: u64 = 1 << 0;
    pub const LOOT: u64 = 1 << 1;
    /// Everything this build prepares and reads from the pack.
    pub const PREPARED: u64 = STRUCTURES | LOOT;

    const NAMES: [(u64, &str); 2] = [(STRUCTURES, "structures"), (LOOT, "loot")];

    /// Names the groups in `bits`, like `structures, loot`.
    #[must_use]
    pub fn names(bits: u64) -> String {
        NAMES
            .iter()
            .filter(|(bit, _)| bits & bit != 0)
            .map(|(_, name)| *name)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

// directory and extension under data/<namespace>/ in the jar
const JAR_SOURCES: [(&str, &str, ResourceKind); 2] = [
    ("structure/", ".nbt", ResourceKind::Structure),
    ("loot_table/", ".json", ResourceKind::LootTable),
];
const MAX_INNER_JAR_LEN: u64 = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Encoding {
    /// Copied from the jar as is.
    Raw = 0,
}

#[derive(Debug, thiserror::Error)]
pub enum PackError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("not a vanilla pack")]
    BadMagic,
    #[error("pack format {found} is not supported (expected {FORMAT_VERSION})")]
    FormatVersion { found: u16 },
    #[error("pack was prepared for data version {found}, server needs {expected}")]
    DataVersion { found: i32, expected: i32 },
    #[error("pack has no {}, which this version of Pumpkin needs", module::names(*.missing))]
    MissingModules { missing: u64 },
    #[error("pack index is corrupt")]
    CorruptIndex,
    #[error("pack entry '{0}' is corrupt")]
    CorruptEntry(String),
    #[error("server jar is not usable: {0}")]
    BadJar(&'static str),
    #[error("server jar is not a valid zip: {0}")]
    Zip(#[from] zip::result::ZipError),
}

pub struct VanillaPack {
    file: File,
    data_version: i32,
    modules: u64,
    source_sha1: [u8; 20],
    entry_count: usize,
    index_len: usize,
    index_hash: u64,
    index: OnceLock<Result<Box<[u8]>, IndexError>>,
}

// cached so every later lookup reports the same failure
#[derive(Clone, Copy, Debug)]
enum IndexError {
    Io(io::ErrorKind),
    Corrupt,
}

impl From<IndexError> for PackError {
    fn from(e: IndexError) -> Self {
        match e {
            IndexError::Io(kind) => Self::Io(kind.into()),
            IndexError::Corrupt => Self::CorruptIndex,
        }
    }
}

impl VanillaPack {
    /// Opens a pack, checking only the header. The index loads on first lookup.
    pub fn open(path: &Path, expected_data_version: i32) -> Result<Self, PackError> {
        let mut file = File::open(path)?;
        let mut header = [0u8; HEADER_LEN];
        file.read_exact(&mut header).map_err(|e| match e.kind() {
            io::ErrorKind::UnexpectedEof => PackError::BadMagic,
            _ => e.into(),
        })?;

        if header[0..8] != MAGIC {
            return Err(PackError::BadMagic);
        }
        let format = u16::from_le_bytes(read_array(&header, 8));
        if format != FORMAT_VERSION {
            return Err(PackError::FormatVersion { found: format });
        }
        let data_version = i32::from_le_bytes(read_array(&header, 12));
        if data_version != expected_data_version {
            return Err(PackError::DataVersion {
                found: data_version,
                expected: expected_data_version,
            });
        }
        let entry_count = u32::from_le_bytes(read_array(&header, 16)) as usize;
        let index_len = u32::from_le_bytes(read_array(&header, 20)) as usize;
        let index_hash = u64::from_le_bytes(read_array(&header, 24));
        if index_len > MAX_INDEX_LEN || entry_count > index_len / RECORD_LEN {
            return Err(PackError::CorruptIndex);
        }

        Ok(Self {
            file,
            data_version,
            modules: u64::from_le_bytes(read_array(&header, 52)),
            source_sha1: read_array(&header, 32),
            entry_count,
            index_len,
            index_hash,
            index: OnceLock::new(),
        })
    }

    /// Loads the index now instead of on the first lookup.
    pub fn load_index(&self) -> Result<(), PackError> {
        self.index()?;
        Ok(())
    }

    fn index(&self) -> Result<&[u8], PackError> {
        let index = self.index.get_or_init(|| {
            let mut index = vec![0u8; self.index_len].into_boxed_slice();
            read_exact_at(&self.file, &mut index, HEADER_LEN as u64)
                .map_err(|e| IndexError::Io(e.kind()))?;
            if xxh64(&index, 0) == self.index_hash {
                Ok(index)
            } else {
                Err(IndexError::Corrupt)
            }
        });
        match index {
            Ok(index) => Ok(index),
            Err(e) => Err((*e).into()),
        }
    }

    #[must_use]
    pub const fn data_version(&self) -> i32 {
        self.data_version
    }

    #[must_use]
    pub const fn modules(&self) -> u64 {
        self.modules
    }

    #[must_use]
    pub const fn source_sha1(&self) -> &[u8; 20] {
        &self.source_sha1
    }

    fn record(index: &[u8], i: usize) -> &[u8] {
        &index[i * RECORD_LEN..(i + 1) * RECORD_LEN]
    }

    // a corrupt offset gives an empty key, which never matches
    fn key<'a>(&self, index: &'a [u8], i: usize) -> &'a [u8] {
        let r = Self::record(index, i);
        let len = u16::from_le_bytes(read_array(r, 2)) as usize;
        let start = self.entry_count * RECORD_LEN + u32::from_le_bytes(read_array(r, 4)) as usize;
        index.get(start..start + len).unwrap_or_default()
    }

    fn find(&self, index: &[u8], kind: ResourceKind, key: &str) -> Option<usize> {
        let (mut lo, mut hi) = (0, self.entry_count);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let probe = (Self::record(index, mid)[0], self.key(index, mid));
            match probe.cmp(&(kind as u8, key.as_bytes())) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    pub fn contains(&self, kind: ResourceKind, key: &str) -> Result<bool, PackError> {
        let index = self.index()?;
        Ok(self.find(index, kind, key).is_some())
    }

    /// Reads one resource by its full id, like `minecraft:igloo/top`.
    pub fn read(&self, kind: ResourceKind, key: &str) -> Result<Option<Vec<u8>>, PackError> {
        let index = self.index()?;
        let Some(i) = self.find(index, kind, key) else {
            return Ok(None);
        };
        let r = Self::record(index, i);
        if r[1] != Encoding::Raw as u8 {
            return Err(PackError::CorruptEntry(key.to_owned()));
        }
        let offset = u64::from_le_bytes(read_array(r, 8));
        let len = u32::from_le_bytes(read_array(r, 16));
        if len > MAX_ENTRY_LEN {
            return Err(PackError::CorruptEntry(key.to_owned()));
        }
        let mut buf = vec![0u8; len as usize];
        read_exact_at(&self.file, &mut buf, offset)?;
        if xxh64(&buf, 0) != u64::from_le_bytes(read_array(r, 24)) {
            return Err(PackError::CorruptEntry(key.to_owned()));
        }
        Ok(Some(buf))
    }

    pub fn keys(&self, kind: ResourceKind) -> Result<impl Iterator<Item = &str>, PackError> {
        let index = self.index()?;
        Ok((0..self.entry_count)
            .filter(move |&i| Self::record(index, i)[0] == kind as u8)
            .filter_map(move |i| std::str::from_utf8(self.key(index, i)).ok()))
    }
}

fn read_array<const N: usize>(bytes: &[u8], at: usize) -> [u8; N] {
    let mut out = [0u8; N];
    out.copy_from_slice(&bytes[at..at + N]);
    out
}

#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[derive(Default)]
pub struct PackWriter {
    entries: Vec<(ResourceKind, String, Vec<u8>)>,
}

impl PackWriter {
    pub fn add(&mut self, kind: ResourceKind, key: String, bytes: Vec<u8>) {
        self.entries.push((kind, key, bytes));
    }

    /// Writes the pack to `path`. Goes through a temp file so a crash can't leave half a pack.
    pub fn write(
        mut self,
        path: &Path,
        data_version: i32,
        modules: u64,
        source_sha1: [u8; 20],
    ) -> Result<(), PackError> {
        self.entries.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        self.entries.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

        let keys_len: usize = self.entries.iter().map(|(_, key, _)| key.len()).sum();
        let index_len = self.entries.len() * RECORD_LEN + keys_len;
        let mut index = Vec::with_capacity(index_len);
        let mut keys = Vec::with_capacity(keys_len);
        let mut offset = (HEADER_LEN + index_len) as u64;
        for (kind, key, bytes) in &self.entries {
            let key_len = u16::try_from(key.len()).map_err(|_| PackError::CorruptIndex)?;
            let key_off = u32::try_from(keys.len()).map_err(|_| PackError::CorruptIndex)?;
            let len = u32::try_from(bytes.len()).map_err(|_| PackError::CorruptIndex)?;
            index.push(*kind as u8);
            index.push(Encoding::Raw as u8);
            index.extend_from_slice(&key_len.to_le_bytes());
            index.extend_from_slice(&key_off.to_le_bytes());
            index.extend_from_slice(&offset.to_le_bytes());
            index.extend_from_slice(&len.to_le_bytes());
            index.extend_from_slice(&[0; 4]);
            index.extend_from_slice(&xxh64(bytes, 0).to_le_bytes());
            keys.extend_from_slice(key.as_bytes());
            offset += u64::from(len);
        }
        index.extend_from_slice(&keys);

        let mut header = [0u8; HEADER_LEN];
        header[0..8].copy_from_slice(&MAGIC);
        header[8..10].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&data_version.to_le_bytes());
        header[16..20].copy_from_slice(&(self.entries.len() as u32).to_le_bytes());
        header[20..24].copy_from_slice(&(index_len as u32).to_le_bytes());
        header[24..32].copy_from_slice(&xxh64(&index, 0).to_le_bytes());
        header[32..52].copy_from_slice(&source_sha1);
        header[52..60].copy_from_slice(&modules.to_le_bytes());

        let tmp = path.with_extension("pak.tmp");
        let mut out = io::BufWriter::new(File::create(&tmp)?);
        out.write_all(&header)?;
        out.write_all(&index)?;
        for (_, _, bytes) in &self.entries {
            out.write_all(bytes)?;
        }
        out.into_inner()
            .map_err(io::IntoInnerError::into_error)?
            .sync_all()?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

/// Builds a pack at `path` from the server download and returns how many resources it wrote.
///
/// The download is a bundler, the real server jar is nested under `META-INF/versions/`.
pub fn build_from_server_jar(
    bundler: &[u8],
    source_sha1: [u8; 20],
    data_version: i32,
    path: &Path,
) -> Result<usize, PackError> {
    let mut bundler = zip::ZipArchive::new(io::Cursor::new(bundler))?;

    // lines look like `<sha256>\t<version id>\t<path under META-INF/versions/>`
    let mut versions = String::new();
    bundler
        .by_name("META-INF/versions.list")?
        .take(64 * 1024)
        .read_to_string(&mut versions)?;
    let mut fields = versions
        .lines()
        .next()
        .ok_or(PackError::BadJar("empty versions.list"))?
        .split('\t');
    let (Some(expected_sha256), Some(_), Some(inner_path)) =
        (fields.next(), fields.next(), fields.next())
    else {
        return Err(PackError::BadJar("malformed versions.list"));
    };

    let mut inner = bundler.by_name(&format!("META-INF/versions/{inner_path}"))?;
    if inner.size() > MAX_INNER_JAR_LEN {
        return Err(PackError::BadJar("nested server jar is too large"));
    }
    let mut inner_bytes = Vec::with_capacity(inner.size() as usize);
    inner.read_to_end(&mut inner_bytes)?;
    drop(inner);
    if hex::encode(Sha256::digest(&inner_bytes)) != expected_sha256 {
        return Err(PackError::BadJar("nested jar checksum mismatch"));
    }

    let mut jar = zip::ZipArchive::new(io::Cursor::new(inner_bytes))?;
    let mut writer = PackWriter::default();
    for i in 0..jar.len() {
        let mut file = jar.by_index(i)?;
        let Some((kind, key)) = jar_resource(file.name()) else {
            continue;
        };
        if file.size() > u64::from(MAX_ENTRY_LEN) {
            return Err(PackError::BadJar("resource is too large"));
        }
        let mut bytes = Vec::with_capacity(file.size() as usize);
        file.read_to_end(&mut bytes)?;
        writer.add(kind, key, bytes);
    }

    let count = writer.entries.len();
    writer.write(path, data_version, module::PREPARED, source_sha1)?;
    Ok(count)
}

// data/minecraft/structure/igloo/top.nbt -> (Structure, minecraft:igloo/top)
fn jar_resource(name: &str) -> Option<(ResourceKind, String)> {
    let (namespace, rest) = name.strip_prefix("data/")?.split_once('/')?;
    JAR_SOURCES.iter().find_map(|(dir, ext, kind)| {
        let path = rest.strip_prefix(dir)?.strip_suffix(ext)?;
        Some((*kind, format!("{namespace}:{path}")))
    })
}

static INSTALLED: OnceLock<VanillaPack> = OnceLock::new();

/// Sets the pack used for vanilla resources. Only the first call wins.
pub fn install(pack: VanillaPack) -> bool {
    INSTALLED.set(pack).is_ok()
}

#[must_use]
pub fn installed() -> Option<&'static VanillaPack> {
    // unit tests never run startup, so pick up the test pack here
    #[cfg(test)]
    install_test_pack();
    INSTALLED.get()
}

/// Installs `PUMPKIN_VANILLA_PAK`, or the default pack name under `target/`, for tests.
///
/// # Panics
///
/// Panics without a usable pack, so tests fail instead of quietly testing nothing.
#[expect(
    clippy::panic,
    reason = "test helper, a missing pack must fail the test"
)]
pub fn install_test_pack() {
    if INSTALLED.get().is_some() {
        return;
    }
    let path = std::env::var_os("PUMPKIN_VANILLA_PAK").map_or_else(
        || {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target")
                .join(pumpkin_config::vanilla_data::VanillaDataConfig::default().pack_path)
        },
        std::path::PathBuf::from,
    );
    match VanillaPack::open(&path, crate::chunk::format::anvil::WORLD_DATA_VERSION) {
        Ok(pack) => {
            install(pack);
        }
        Err(e) => panic!(
            "this test needs vanilla data, but {} is not usable ({e}). Prepare a pack there or set PUMPKIN_VANILLA_PAK",
            path.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_corruption() {
        let dir = std::env::temp_dir().join(format!("pumpkin-pack-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vanilla.pak");

        let mut writer = PackWriter::default();
        writer.add(
            ResourceKind::LootTable,
            "minecraft:b".into(),
            b"loot".to_vec(),
        );
        writer.add(
            ResourceKind::Structure,
            "minecraft:z".into(),
            b"zz".to_vec(),
        );
        writer.add(
            ResourceKind::Structure,
            "minecraft:a".into(),
            b"aaa".to_vec(),
        );
        writer.write(&path, 42, module::PREPARED, [7; 20]).unwrap();

        assert!(matches!(
            VanillaPack::open(&path, 43),
            Err(PackError::DataVersion { found: 42, .. })
        ));
        let pack = VanillaPack::open(&path, 42).unwrap();
        assert_eq!(pack.modules(), module::PREPARED);
        assert_eq!(
            pack.read(ResourceKind::Structure, "minecraft:a")
                .unwrap()
                .as_deref(),
            Some(&b"aaa"[..])
        );
        assert_eq!(
            pack.read(ResourceKind::Structure, "minecraft:b").unwrap(),
            None
        );
        assert_eq!(
            pack.keys(ResourceKind::Structure)
                .unwrap()
                .collect::<Vec<_>>(),
            ["minecraft:a", "minecraft:z"]
        );

        // kind sorts first, so the last blob is the loot table
        let mut bytes = std::fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        let pack = VanillaPack::open(&path, 42).unwrap();
        assert!(pack.read(ResourceKind::Structure, "minecraft:z").is_ok());
        assert!(matches!(
            pack.read(ResourceKind::LootTable, "minecraft:b"),
            Err(PackError::CorruptEntry(_))
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

use std::{
    collections::BTreeSet,
    env,
    ffi::OsString,
    fs::File,
    io::{self, Write},
    path::PathBuf,
    process::Command,
};

const GRADIENTS: [[f32; 3]; 16] = include!("../pumpkin-util/src/noise/perlin_gradients.in");

/// RTX consumer GPUs and major datacenter/server GPUs:
/// - 75: Turing (RTX 20xx, T4)
/// - 80: Ampere DC (A100, A30)
/// - 86: Ampere (RTX 30xx, A10, A40, RTX A6000)
/// - 89: Ada Lovelace (RTX 40xx, L4, L40, L40S)
/// - 90: Hopper (H100, H200)
/// - 100: Blackwell DC (B100, B200)
/// - 120: Blackwell (RTX 50xx)
const ALL_ARCHES: &[u32] = &[75, 80, 86, 89, 90, 100, 120];

enum ArchSource {
    Detected,
    All,
    Explicit,
}

enum ArchPlan {
    Sass {
        source: ArchSource,
        arches: Vec<u32>,
    },
    PtxOnly(u32),
}

fn main() -> io::Result<()> {
    let cuda_dir = "src/generation/noise/gpu/cuda/kernels";
    let vulkan_dir = "src/generation/noise/gpu/vulkan";
    println!("cargo:rerun-if-changed={cuda_dir}/density.cuh");
    println!("cargo:rerun-if-changed={cuda_dir}/worldgen.cu");
    println!("cargo:rerun-if-changed={vulkan_dir}/worldgen.comp");
    println!("cargo:rerun-if-changed=../pumpkin-util/src/noise/perlin_gradients.in");

    let build_cuda = env::var_os("CARGO_FEATURE_CUDA_WORLDGEN").is_some();
    let build_vulkan = env::var_os("CARGO_FEATURE_VULKAN_WORLDGEN").is_some();
    if !build_cuda && !build_vulkan {
        return Ok(());
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").ok_or(io::ErrorKind::NotFound)?);

    if build_cuda {
        compile_cuda(&out_dir, cuda_dir)?;
    }
    if build_vulkan {
        compile_vulkan(&out_dir, vulkan_dir)?;
    }
    Ok(())
}

fn compile_cuda(out_dir: &PathBuf, dir: &str) -> io::Result<()> {
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=PUMPKIN_CUDA_ARCH");

    let mut header = File::create(out_dir.join("perlin_gradients.cuh"))?;
    writeln!(
        header,
        "__device__ __constant__ float gradients[{}][3] = {{",
        GRADIENTS.len()
    )?;
    for [x, y, z] in GRADIENTS {
        writeln!(header, "    {{{x:?}f, {y:?}f, {z:?}f}},")?;
    }
    writeln!(header, "}};")?;

    let nvcc = env::var_os("NVCC").unwrap_or_else(|| OsString::from("nvcc"));
    let supported = query_supported_arches(&nvcc)?;
    let plan = resolve_arch_plan(&supported)?;

    let mut command = Command::new(&nvcc);
    command.args(["--fatbin", "-std=c++17", "--fmad=false"]);
    match plan {
        ArchPlan::Sass { source, arches } => {
            let list = format_arches(&arches);
            if let Some(&highest) = arches.last() {
                match source {
                    ArchSource::Detected => println!(
                        "cargo:warning=CUDA worldgen: detected local GPU arch(es) {list}; building fatbin (+ compute_{highest} PTX)"
                    ),
                    ArchSource::All => println!(
                        "cargo:warning=CUDA worldgen: building fatbin for all supported RTX/server arches: {list} (+ compute_{highest} PTX)"
                    ),
                    ArchSource::Explicit => println!(
                        "cargo:warning=CUDA worldgen: building fatbin for requested arch(es): {list} (+ compute_{highest} PTX)"
                    ),
                }
            }
            for &arch in &arches {
                command.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
            }
            if let Some(&highest) = arches.last() {
                command.arg(format!(
                    "-gencode=arch=compute_{highest},code=compute_{highest}"
                ));
            }
        }
        ArchPlan::PtxOnly(arch) => {
            command.arg(format!("-gencode=arch=compute_{arch},code=compute_{arch}"));
        }
    }

    let status = command
        .arg("-I")
        .arg(out_dir)
        .arg(format!("{dir}/worldgen.cu"))
        .arg("-o")
        .arg(out_dir.join("worldgen.fatbin"))
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other("nvcc failed to build worldgen.fatbin"))
    }
}

fn compile_vulkan(out_dir: &PathBuf, dir: &str) -> io::Result<()> {
    println!("cargo:rerun-if-env-changed=GLSLC");

    let mut header = File::create(out_dir.join("perlin_gradients.glsl"))?;
    writeln!(
        header,
        "const vec3 GRADIENTS[{}] = vec3[{}](",
        GRADIENTS.len(),
        GRADIENTS.len()
    )?;
    for (index, [x, y, z]) in GRADIENTS.iter().enumerate() {
        let comma = if index + 1 < GRADIENTS.len() { "," } else { "" };
        writeln!(header, "    vec3({x:?}, {y:?}, {z:?}){comma}")?;
    }
    writeln!(header, ");")?;

    let shader = format!("{dir}/worldgen.comp");
    let output = out_dir.join("worldgen.spv");

    let glslc = env::var_os("GLSLC").unwrap_or_else(|| OsString::from("glslc"));
    let status = Command::new(&glslc)
        .args(["--target-env=vulkan1.2", "-O", "-I"])
        .arg(out_dir)
        .arg(&shader)
        .arg("-o")
        .arg(&output)
        .status();

    match status {
        Ok(status) if status.success() => Ok(()),
        Ok(_) => Err(io::Error::other("glslc failed to build worldgen.spv")),
        Err(error) if error.kind() == io::ErrorKind::NotFound && env::var_os("GLSLC").is_none() => {
            let fallback = Command::new("glslangValidator")
                .args(["-V", "--target-env", "vulkan1.2", "-I"])
                .arg(out_dir)
                .arg(&shader)
                .arg("-o")
                .arg(&output)
                .status()?;
            if fallback.success() {
                Ok(())
            } else {
                Err(io::Error::other(
                    "glslangValidator failed to build worldgen.spv",
                ))
            }
        }
        Err(error) => Err(error),
    }
}

fn query_supported_arches(nvcc: &OsString) -> io::Result<Vec<u32>> {
    let output = Command::new(nvcc).arg("--list-gpu-arch").output()?;
    if !output.status.success() {
        return Err(io::Error::other("nvcc --list-gpu-arch failed"));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut arches = BTreeSet::new();
    for line in stdout.lines() {
        if let Some(rest) = line.trim().strip_prefix("compute_")
            && let Ok(arch) = rest.parse::<u32>()
        {
            arches.insert(arch);
        }
    }
    if arches.is_empty() {
        return Err(io::Error::other(
            "nvcc --list-gpu-arch returned no supported architectures",
        ));
    }
    Ok(arches.into_iter().collect())
}

fn resolve_arch_plan(supported: &[u32]) -> io::Result<ArchPlan> {
    let requested = env::var("PUMPKIN_CUDA_ARCH").ok();
    let requested = requested
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());

    match requested {
        Some(spec) if spec.eq_ignore_ascii_case("all") => all_arches(supported),
        Some(spec) if spec.eq_ignore_ascii_case("native") => detect_or_fallback(supported),
        Some(spec) => parse_explicit_arches(spec, supported),
        None if env::var_os("CARGO_FEATURE_CUDA_WORLDGEN_ALL").is_some() => all_arches(supported),
        None => detect_or_fallback(supported),
    }
}

fn all_arches(supported: &[u32]) -> io::Result<ArchPlan> {
    let arches: Vec<u32> = ALL_ARCHES
        .iter()
        .copied()
        .filter(|arch| supported.contains(arch))
        .collect();
    if arches.is_empty() {
        return Err(io::Error::other(format!(
            "none of the predefined CUDA architectures are supported by this nvcc (supported: {supported:?})"
        )));
    }
    Ok(ArchPlan::Sass {
        source: ArchSource::All,
        arches,
    })
}

fn parse_explicit_arches(spec: &str, supported: &[u32]) -> io::Result<ArchPlan> {
    let mut arches = BTreeSet::new();
    for token in spec.split([',', ';']) {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let arch = parse_arch_token(token).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid PUMPKIN_CUDA_ARCH entry {token:?}"),
            )
        })?;
        if !supported.contains(&arch) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "CUDA architecture {arch} is not supported by this nvcc (supported: {supported:?})"
                ),
            ));
        }
        arches.insert(arch);
    }
    if arches.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "PUMPKIN_CUDA_ARCH did not contain any architectures",
        ));
    }
    Ok(ArchPlan::Sass {
        source: ArchSource::Explicit,
        arches: arches.into_iter().collect(),
    })
}

fn detect_or_fallback(supported: &[u32]) -> io::Result<ArchPlan> {
    if let Some(detected) = detect_host_gpu_arches() {
        let mut selected = BTreeSet::new();
        for cap in detected {
            if supported.contains(&cap) {
                selected.insert(cap);
            } else if let Some(&fallback) = supported.iter().rfind(|&&arch| arch <= cap) {
                selected.insert(fallback);
            } else {
                return Err(io::Error::other(format!(
                    "detected GPU compute capability {cap} is older than nvcc's minimum supported architecture ({supported:?})"
                )));
            }
        }
        if !selected.is_empty() {
            return Ok(ArchPlan::Sass {
                source: ArchSource::Detected,
                arches: selected.into_iter().collect(),
            });
        }
    }

    let min_arch = supported
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("no supported CUDA architectures"))?;
    println!(
        "cargo:warning=CUDA worldgen: could not detect local NVIDIA GPU via nvidia-smi; building PTX-only fallback for compute_{min_arch} (set PUMPKIN_CUDA_ARCH=all or PUMPKIN_CUDA_ARCH=<sm> to target specific GPUs)"
    );
    Ok(ArchPlan::PtxOnly(min_arch))
}

fn format_arches(arches: &[u32]) -> String {
    arches
        .iter()
        .map(|arch| format!("sm_{arch}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn detect_host_gpu_arches() -> Option<Vec<u32>> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut caps = BTreeSet::new();
    for line in stdout.lines() {
        if let Some(arch) = parse_arch_token(line.trim()) {
            caps.insert(arch);
        }
    }
    (!caps.is_empty()).then(|| caps.into_iter().collect())
}

fn parse_arch_token(token: &str) -> Option<u32> {
    let stripped = token
        .strip_prefix("sm_")
        .or_else(|| token.strip_prefix("compute_"))
        .unwrap_or(token);
    if let Some((major, minor)) = stripped.split_once('.') {
        let major = major.trim().parse::<u32>().ok()?;
        let minor = minor.trim().parse::<u32>().ok()?;
        if minor < 10 {
            return major.checked_mul(10)?.checked_add(minor);
        }
        return None;
    }
    stripped.parse::<u32>().ok()
}

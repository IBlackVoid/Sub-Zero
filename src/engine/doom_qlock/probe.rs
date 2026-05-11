use super::util::{find_in_path, sanitize_fingerprint_component};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug, Clone)]
pub(super) struct HardwareProbe {
    pub(super) cpu_cores: usize,
    pub(super) total_ram_mb: Option<u64>,
    pub(super) disk_write_mbps: Option<f64>,
    pub(super) gpu: Option<GpuProbe>,
}

impl HardwareProbe {
    pub(super) fn probe() -> Self {
        Self {
            cpu_cores: std::thread::available_parallelism()
                .map(|cores| cores.get())
                .unwrap_or(4),
            total_ram_mb: probe_total_ram_mb(),
            disk_write_mbps: probe_disk_write_mbps(),
            gpu: probe_gpu(),
        }
    }

    pub(super) fn fingerprint(&self) -> String {
        let gpu_component = self
            .gpu
            .as_ref()
            .map(|gpu| {
                format!(
                    "{}-{}",
                    sanitize_fingerprint_component(&gpu.backend),
                    sanitize_fingerprint_component(&gpu.name)
                )
            })
            .unwrap_or_else(|| "none".to_string());
        let gpu_vram = self.gpu.as_ref().and_then(|gpu| gpu.vram_mb).unwrap_or(0);
        let disk_mbps = self.disk_write_mbps.unwrap_or(0.0).round() as u64;
        format!(
            "cpu{}-ram{}-gpu{}-vram{}-disk{}",
            self.cpu_cores,
            self.total_ram_mb.unwrap_or(0),
            gpu_component,
            gpu_vram,
            disk_mbps
        )
    }

    pub(super) fn gpu_summary(&self) -> String {
        let Some(gpu) = &self.gpu else {
            return "none".to_string();
        };
        let vram = gpu
            .vram_mb
            .map(|value| format!("{value}MB"))
            .unwrap_or_else(|| "unknown-vram".to_string());
        let cc = gpu
            .compute_capability
            .as_ref()
            .map(|value| format!("cc={value}"))
            .unwrap_or_else(|| "cc=unknown".to_string());
        format!("{}:{} ({}, {})", gpu.backend, gpu.name, vram, cc)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct GpuProbe {
    pub(super) backend: String,
    pub(super) name: String,
    pub(super) vram_mb: Option<u64>,
    pub(super) compute_capability: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct HardwareSnapshot {
    pub(super) cpu_cores: usize,
    pub(super) total_ram_mb: Option<u64>,
    pub(super) disk_write_mbps: Option<f64>,
    pub(super) gpu_backend: Option<String>,
    pub(super) gpu_vram_mb: Option<u64>,
}

impl From<&HardwareProbe> for HardwareSnapshot {
    fn from(value: &HardwareProbe) -> Self {
        Self {
            cpu_cores: value.cpu_cores,
            total_ram_mb: value.total_ram_mb,
            disk_write_mbps: value.disk_write_mbps,
            gpu_backend: value
                .gpu
                .as_ref()
                .map(|gpu| gpu.backend.to_ascii_lowercase()),
            gpu_vram_mb: value.gpu.as_ref().and_then(|gpu| gpu.vram_mb),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct MediaProbe {
    pub(super) format_name: Option<String>,
    pub(super) duration_secs: Option<f64>,
}

pub(super) fn probe_media_format(input: &Path) -> Result<MediaProbe, String> {
    let ffprobe = find_in_path(&["ffprobe", "ffprobe.exe"])
        .ok_or_else(|| "ffprobe not found in PATH".to_string())?;

    let output = Command::new(ffprobe)
        .arg("-hide_banner")
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("format=duration,format_name")
        .arg("-of")
        .arg("default=noprint_wrappers=1:nokey=0")
        .arg(input)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| format!("failed to spawn ffprobe: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "ffprobe failed with status {}: {}",
            output.status,
            stderr.trim()
        ));
    }

    parse_ffprobe_output(&String::from_utf8_lossy(&output.stdout))
}

pub(super) fn parse_ffprobe_output(stdout: &str) -> Result<MediaProbe, String> {
    let mut probe = MediaProbe::default();
    for line in stdout.lines() {
        let trimmed = line.trim();
        if let Some(value) = trimmed.strip_prefix("format_name=") {
            if !value.is_empty() {
                probe.format_name = Some(value.to_string());
            }
            continue;
        }
        if let Some(value) = trimmed.strip_prefix("duration=") {
            let duration = value
                .parse::<f64>()
                .map_err(|_| format!("invalid ffprobe duration: {value}"))?;
            if duration.is_finite() && duration > 0.0 {
                probe.duration_secs = Some(duration);
            }
        }
    }
    Ok(probe)
}

fn probe_total_ram_mb() -> Option<u64> {
    if cfg!(target_os = "linux") {
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if let Some(value) = line.strip_prefix("MemTotal:") {
                    let kb = value
                        .split_whitespace()
                        .next()
                        .and_then(|chunk| chunk.parse::<u64>().ok())?;
                    return Some(kb / 1024);
                }
            }
        }
    }

    if cfg!(target_os = "windows") {
        let output = Command::new("wmic")
            .arg("ComputerSystem")
            .arg("get")
            .arg("TotalPhysicalMemory")
            .arg("/value")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .ok()?;
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if let Some(value) = line.trim().strip_prefix("TotalPhysicalMemory=") {
                    let bytes = value.trim().parse::<u64>().ok()?;
                    return Some(bytes / 1024 / 1024);
                }
            }
        }
    }

    None
}

fn probe_gpu() -> Option<GpuProbe> {
    probe_nvidia_gpu()
        .or_else(probe_rocm_gpu)
        .or_else(probe_metal_gpu)
}

fn probe_nvidia_gpu() -> Option<GpuProbe> {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total,compute_cap")
        .arg("--format=csv,noheader,nounits")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout.lines().find(|line| !line.trim().is_empty())?;
    let parts: Vec<&str> = first_line.split(',').map(str::trim).collect();
    let name = parts.first()?.to_string();
    let vram_mb = parts.get(1).and_then(|value| value.parse::<u64>().ok());
    let compute_capability = parts.get(2).map(|value| value.to_string());

    Some(GpuProbe {
        backend: "cuda".to_string(),
        name,
        vram_mb,
        compute_capability,
    })
}

fn probe_rocm_gpu() -> Option<GpuProbe> {
    let rocm_smi = find_in_path(&["rocm-smi", "rocm-smi.exe"])?;
    let output = Command::new(rocm_smi)
        .arg("--showproductname")
        .arg("--showmeminfo")
        .arg("vram")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut name = None::<String>;
    let mut vram_mb = None::<u64>;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if name.is_none()
            && (trimmed.to_ascii_lowercase().contains("card series")
                || trimmed.to_ascii_lowercase().contains("product name"))
        {
            name = trimmed
                .split(':')
                .nth(1)
                .map(str::trim)
                .map(ToString::to_string)
                .filter(|value| !value.is_empty());
        }
        if vram_mb.is_none() && trimmed.to_ascii_lowercase().contains("total") {
            let mb = trimmed
                .split_whitespace()
                .find_map(|part| part.parse::<u64>().ok())
                .map(|kb_or_mb| {
                    if kb_or_mb > 200_000 {
                        kb_or_mb / 1024
                    } else {
                        kb_or_mb
                    }
                });
            vram_mb = mb;
        }
    }
    let name = name.unwrap_or_else(|| "AMD GPU".to_string());
    Some(GpuProbe {
        backend: "rocm".to_string(),
        name,
        vram_mb,
        compute_capability: None,
    })
}

fn probe_metal_gpu() -> Option<GpuProbe> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let output = Command::new("system_profiler")
        .arg("SPDisplaysDataType")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut chip = None::<String>;
    let mut vram_mb = None::<u64>;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if chip.is_none() && trimmed.starts_with("Chipset Model:") {
            chip = trimmed
                .split_once(':')
                .map(|(_, value)| value.trim().to_string())
                .filter(|value| !value.is_empty());
        }
        if vram_mb.is_none() && trimmed.starts_with("VRAM") {
            let parsed = trimmed
                .split_whitespace()
                .find_map(|part| part.parse::<u64>().ok())
                .map(|value| {
                    if trimmed.to_ascii_lowercase().contains("gb") {
                        value * 1024
                    } else {
                        value
                    }
                });
            vram_mb = parsed;
        }
    }
    let name = chip.unwrap_or_else(|| "Apple GPU".to_string());
    Some(GpuProbe {
        backend: "metal".to_string(),
        name,
        vram_mb,
        compute_capability: None,
    })
}

fn probe_disk_write_mbps() -> Option<f64> {
    let root = std::env::var_os("SUB_ZERO_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
        .unwrap_or_else(std::env::temp_dir);
    let probe_dir = root.join(".sub-zero");
    if fs::create_dir_all(&probe_dir).is_err() {
        return None;
    }
    let probe_file = probe_dir.join(".disk_probe.bin");
    let payload = vec![0u8; 1_048_576];
    let start = std::time::Instant::now();
    if fs::write(&probe_file, &payload).is_err() {
        return None;
    }
    let elapsed = start.elapsed().as_secs_f64().max(0.000_5);
    let _ = fs::remove_file(&probe_file);
    Some((payload.len() as f64 / 1_048_576.0) / elapsed)
}

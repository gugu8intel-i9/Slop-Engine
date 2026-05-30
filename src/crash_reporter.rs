// src/crash_reporter.rs
// Collect stack traces, GPU info, logs on crash

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use chrono::Local;

static CRASH_REPORTER_INITIALIZED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone)]
pub struct CrashInfo {
    pub timestamp: String,
    pub error_message: String,
    pub stack_trace: String,
    pub gpu_info: GpuInfo,
    pub system_info: SystemInfo,
    pub recent_logs: Vec<String>,
    pub memory_info: MemoryInfo,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub vendor: String,
    pub renderer: String,
    pub version: String,
    pub vram_mb: u32,
}

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub os: String,
    pub cpu: String,
    pub ram_gb: u32,
    pub cores: u32,
}

#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub used_mb: u32,
    pub free_mb: u32,
    pub total_mb: u32,
}

pub struct CrashReporter {
    report_dir: PathBuf,
    enabled: bool,
}

impl CrashReporter {
    pub fn new<P: AsRef<Path>>(report_dir: P) -> Self {
        Self {
            report_dir: report_dir.as_ref().to_path_buf(),
            enabled: true,
        }
    }

    pub fn initialize(&self) -> Result<(), std::io::Error> {
        if CRASH_REPORTER_INITIALIZED.swap(true, Ordering::SeqCst) {
            return Ok(());
        }

        std::fs::create_dir_all(&self.report_dir)?;
        
        #[cfg(unix)]
        self.register_signal_handlers();
        
        #[cfg(windows)]
        self.register_exception_handler();

        Ok(())
    }

    #[cfg(unix)]
    fn register_signal_handlers(&self) {
        unsafe {
            libc::signal(libc::SIGSEGV, handle_crash as libc::sighandler_t);
            libc::signal(libc::SIGABRT, handle_crash as libc::sighandler_t);
            libc::signal(libc::SIGILL, handle_crash as libc::sighandler_t);
            libc::signal(libc::SIGFPE, handle_crash as libc::sighandler_t);
        }
    }

    #[cfg(windows)]
    fn register_exception_handler(&self) {
        // Windows exception handler registration would go here
    }

    pub fn generate_report(&self, error: &str, stack_trace: &str) -> Result<PathBuf, std::io::Error> {
        let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
        let filename = format!("crash_{}.txt", timestamp);
        let filepath = self.report_dir.join(&filename);

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&filepath)?;

        let gpu_info = self.collect_gpu_info();
        let system_info = self.collect_system_info();
        let memory_info = self.collect_memory_info();
        let logs = self.collect_recent_logs();

        writeln!(file, "=== CRASH REPORT ===")?;
        writeln!(file, "Timestamp: {}", timestamp)?;
        writeln!(file, "Error: {}\n", error)?;
        
        writeln!(file, "=== STACK TRACE ===")?;
        writeln!(file, "{}\n", stack_trace)?;
        
        writeln!(file, "=== GPU INFO ===")?;
        writeln!(file, "Vendor: {}", gpu_info.vendor)?;
        writeln!(file, "Renderer: {}", gpu_info.renderer)?;
        writeln!(file, "Version: {}", gpu_info.version)?;
        writeln!(file, "VRAM: {} MB\n", gpu_info.vram_mb)?;
        
        writeln!(file, "=== SYSTEM INFO ===")?;
        writeln!(file, "OS: {}", system_info.os)?;
        writeln!(file, "CPU: {}", system_info.cpu)?;
        writeln!(file, "RAM: {} GB", system_info.ram_gb)?;
        writeln!(file, "Cores: {}\n", system_info.cores)?;
        
        writeln!(file, "=== MEMORY INFO ===")?;
        writeln!(file, "Used: {} MB", memory_info.used_mb)?;
        writeln!(file, "Free: {} MB", memory_info.free_mb)?;
        writeln!(file, "Total: {} MB\n", memory_info.total_mb)?;
        
        if !logs.is_empty() {
            writeln!(file, "=== RECENT LOGS ===")?;
            for log in logs {
                writeln!(file, "{}", log)?;
            }
        }

        Ok(filepath)
    }

    fn collect_gpu_info(&self) -> GpuInfo {
        // In a real implementation, this would query wgpu/dx/vulkan
        GpuInfo {
            vendor: "Unknown".to_string(),
            renderer: "Unknown".to_string(),
            version: "Unknown".to_string(),
            vram_mb: 0,
        }
    }

    fn collect_system_info(&self) -> SystemInfo {
        SystemInfo {
            os: std::env::consts::OS.to_string(),
            cpu: "Unknown".to_string(),
            ram_gb: sysinfo::System::new_all().total_memory() / 1024 / 1024 / 1024,
            cores: num_cpus::get() as u32,
        }
    }

    fn collect_memory_info(&self) -> MemoryInfo {
        let sys = sysinfo::System::new_all();
        MemoryInfo {
            used_mb: (sys.used_memory() / 1024 / 1024) as u32,
            free_mb: (sys.free_memory() / 1024 / 1024) as u32,
            total_mb: (sys.total_memory() / 1024 / 1024) as u32,
        }
    }

    fn collect_recent_logs(&self) -> Vec<String> {
        // Would integrate with logger module
        Vec::new()
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

#[cfg(unix)]
extern "C" fn handle_crash(sig: libc::c_int) {
    eprintln!("Crash detected! Signal: {}", sig);
    // Generate minimal crash report
    std::process::abort();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crash_reporter_creation() {
        let reporter = CrashReporter::new("/tmp/crash_reports");
        assert!(reporter.enabled);
    }
}

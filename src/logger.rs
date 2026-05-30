// src/logger.rs
// Structured logging, log levels, file output

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;
use chrono::{DateTime, Local};

/// Log level severity
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Fatal = 5,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
            LogLevel::Fatal => "FATAL",
        }
    }
}

/// Log entry structure
#[derive(Clone, Debug)]
pub struct LogEntry {
    pub timestamp: DateTime<Local>,
    pub level: LogLevel,
    pub target: String,
    pub message: String,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub fields: Vec<(String, String)>,
}

impl LogEntry {
    pub fn format(&self, colored: bool) -> String {
        let time_str = self.timestamp.format("%H:%M:%S%.3f").to_string();
        
        let level_color = if colored {
            match self.level {
                LogLevel::Trace => "\x1b[90m",   // Bright black
                LogLevel::Debug => "\x1b[36m",   // Cyan
                LogLevel::Info => "\x1b[32m",    // Green
                LogLevel::Warn => "\x1b[33m",    // Yellow
                LogLevel::Error => "\x1b[31m",   // Red
                LogLevel::Fatal => "\x1b[91m",   // Bright red
            }
        } else {
            ""
        };
        
        let reset = if colored { "\x1b[0m" } else { "" };
        
        let mut output = format!(
            "{} [{}{}{}] {}: ",
            time_str,
            level_color,
            self.level.as_str(),
            reset,
            self.target
        );
        
        output.push_str(&self.message);
        
        if !self.fields.is_empty() {
            output.push_str(" {");
            let field_strs: Vec<String> = self.fields.iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            output.push_str(&field_strs.join(", "));
            output.push('}');
        }
        
        if let (Some(file), Some(line)) = (&self.file, self.line) {
            output.push_str(&format!(" ({}:{})", file, line));
        }
        
        output
    }
}

/// Log writer trait for multiple outputs
pub trait LogWriter: Send + Sync {
    fn write(&self, entry: &LogEntry) -> std::io::Result<()>;
    fn flush(&self) -> std::io::Result<()>;
}

/// Console log writer
pub struct ConsoleWriter {
    use_colors: bool,
}

impl ConsoleWriter {
    pub fn new(use_colors: bool) -> Self {
        Self { use_colors }
    }
}

impl LogWriter for ConsoleWriter {
    fn write(&self, entry: &LogEntry) -> std::io::Result<()> {
        println!("{}", entry.format(self.use_colors));
        Ok(())
    }

    fn flush(&self) -> std::io::Result<()> {
        std::io::stdout().flush()
    }
}

/// File log writer
pub struct FileWriter {
    file: RwLock<BufWriter<File>>,
    path: PathBuf,
}

impl FileWriter {
    pub fn new<P: AsRef<Path>>(path: P, append: bool) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .append(append)
            .open(&path)?;

        Ok(Self {
            file: RwLock::new(BufWriter::new(file)),
            path,
        })
    }
}

impl LogWriter for FileWriter {
    fn write(&self, entry: &LogEntry) -> std::io::Result<()> {
        let line = entry.format(false) + "\n";
        let mut file = self.file.write();
        file.write_all(line.as_bytes())?;
        Ok(())
    }

    fn flush(&self) -> std::io::Result<()> {
        self.file.write().flush()
    }
}

/// Logger configuration
#[derive(Clone, Debug)]
pub struct LoggerConfig {
    pub max_level: LogLevel,
    pub console_output: bool,
    pub console_colors: bool,
    pub file_output: bool,
    pub file_path: Option<PathBuf>,
    pub file_rotation: bool,
    pub max_file_size_mb: u64,
    pub include_timestamp: bool,
    pub include_location: bool,
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self {
            max_level: LogLevel::Info,
            console_output: true,
            console_colors: true,
            file_output: false,
            file_path: None,
            file_rotation: true,
            max_file_size_mb: 10,
            include_timestamp: true,
            include_location: false,
        }
    }
}

/// Main logger instance
pub struct Logger {
    config: RwLock<LoggerConfig>,
    writers: RwLock<Vec<Arc<dyn LogWriter>>>,
    buffer: RwLock<Vec<LogEntry>>,
    buffer_size: usize,
}

impl Logger {
    pub fn new(config: LoggerConfig) -> Result<Self, std::io::Error> {
        let mut writers: Vec<Arc<dyn LogWriter>> = Vec::new();

        if config.console_output {
            writers.push(Arc::new(ConsoleWriter::new(config.console_colors)));
        }

        if config.file_output {
            if let Some(ref path) = config.file_path {
                writers.push(Arc::new(FileWriter::new(path, true)?));
            }
        }

        Ok(Self {
            config: RwLock::new(config),
            writers: RwLock::new(writers),
            buffer: RwLock::new(Vec::new()),
            buffer_size: 1000,
        })
    }

    pub fn log(&self, level: LogLevel, target: &str, message: &str) {
        let config = self.config.read();
        
        if level < config.max_level {
            return;
        }

        let entry = LogEntry {
            timestamp: Local::now(),
            level,
            target: target.to_string(),
            message: message.to_string(),
            file: None,
            line: None,
            fields: Vec::new(),
        };

        drop(config);
        self.write_entry(entry);
    }

    pub fn log_with_fields(
        &self,
        level: LogLevel,
        target: &str,
        message: &str,
        fields: Vec<(String, String)>,
    ) {
        let config = self.config.read();
        
        if level < config.max_level {
            return;
        }

        let entry = LogEntry {
            timestamp: Local::now(),
            level,
            target: target.to_string(),
            message: message.to_string(),
            file: None,
            line: None,
            fields,
        };

        drop(config);
        self.write_entry(entry);
    }

    fn write_entry(&self, entry: LogEntry) {
        // Add to buffer
        {
            let mut buffer = self.buffer.write();
            buffer.push(entry.clone());
            
            if buffer.len() > self.buffer_size {
                buffer.remove(0);
            }
        }

        // Write to all writers
        let writers = self.writers.read();
        for writer in writers.iter() {
            let _ = writer.write(&entry);
        }
    }

    pub fn trace(&self, target: &str, message: &str) {
        self.log(LogLevel::Trace, target, message);
    }

    pub fn debug(&self, target: &str, message: &str) {
        self.log(LogLevel::Debug, target, message);
    }

    pub fn info(&self, target: &str, message: &str) {
        self.log(LogLevel::Info, target, message);
    }

    pub fn warn(&self, target: &str, message: &str) {
        self.log(LogLevel::Warn, target, message);
    }

    pub fn error(&self, target: &str, message: &str) {
        self.log(LogLevel::Error, target, message);
    }

    pub fn fatal(&self, target: &str, message: &str) {
        self.log(LogLevel::Fatal, target, message);
    }

    pub fn set_level(&self, level: LogLevel) {
        self.config.write().max_level = level;
    }

    pub fn add_writer(&self, writer: Arc<dyn LogWriter>) {
        self.writers.write().push(writer);
    }

    pub fn flush(&self) {
        let writers = self.writers.read();
        for writer in writers.iter() {
            let _ = writer.flush();
        }
    }

    pub fn get_recent_logs(&self, count: usize) -> Vec<LogEntry> {
        let buffer = self.buffer.read();
        buffer.iter().rev().take(count).cloned().collect()
    }
}

/// Global logger instance
static GLOBAL_LOGGER: parking_lot::OnceCell<Logger> = parking_lot::OnceCell::new();

pub fn init_logger(config: LoggerConfig) -> Result<(), std::io::Error> {
    let logger = Logger::new(config)?;
    GLOBAL_LOGGER.set(logger).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::Other, "Logger already initialized")
    })?;
    Ok(())
}

pub fn get_logger() -> Option<&'static Logger> {
    GLOBAL_LOGGER.get()
}

/// Convenience macros
#[macro_export]
macro_rules! log_trace {
    ($target:expr, $($arg:tt)*) => {
        if let Some(logger) = $crate::logger::get_logger() {
            logger.trace($target, &format!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! log_debug {
    ($target:expr, $($arg:tt)*) => {
        if let Some(logger) = $crate::logger::get_logger() {
            logger.debug($target, &format!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! log_info {
    ($target:expr, $($arg:tt)*) => {
        if let Some(logger) = $crate::logger::get_logger() {
            logger.info($target, &format!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! log_warn {
    ($target:expr, $($arg:tt)*) => {
        if let Some(logger) = $crate::logger::get_logger() {
            logger.warn($target, &format!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! log_error {
    ($target:expr, $($arg:tt)*) => {
        if let Some(logger) = $crate::logger::get_logger() {
            logger.error($target, &format!($($arg)*));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Error > LogLevel::Warn);
        assert!(LogLevel::Warn > LogLevel::Info);
        assert!(LogLevel::Info > LogLevel::Debug);
        assert!(LogLevel::Debug > LogLevel::Trace);
    }

    #[test]
    fn test_logger_creation() {
        let config = LoggerConfig {
            console_output: false,
            file_output: false,
            ..Default::default()
        };
        
        let logger = Logger::new(config).unwrap();
        assert_eq!(logger.config.read().max_level, LogLevel::Info);
    }
}

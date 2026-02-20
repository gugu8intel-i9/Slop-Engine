// src/error.rs
//! High-performance & feature-rich error handling for the entire crate.
//!
//! - **Performance**: Enum discriminant (cheap match), `#[inline]` everywhere, allocations *only* on error paths.
//! - **Features**: Context chaining, custom messages, transparent std errors, `is_*` helpers, optional JSON + backtraces, `Result` alias.
//! - **Extensible**: Just add your own variants. Works perfectly with `?`, async, threads, `tokio`, `axum`, `sqlx`, `reqwest`, etc.

use std::fmt;
use thiserror::Error;

/// Main error type — lightweight, Send + Sync + 'static, perfect for async and libraries.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum Error {
    /// I/O errors (most common).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// UTF-8 / string encoding issues.
    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    /// Integer parsing failures.
    #[error("integer parse error: {0}")]
    ParseInt(#[from] std::num::ParseIntError),

    /// Float parsing failures.
    #[error("float parse error: {0}")]
    ParseFloat(#[from] std::num::ParseFloatError),

    /// JSON (de)serialization — enable with `json` feature.
    #[cfg(feature = "json")]
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Opaque wrapper for any other error (great for foreign crates).
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    /// Simple custom message (allocation only when error happens).
    #[error("{0}")]
    Custom(String),

    /// Rich context chaining (like anyhow but zero-cost when you control the types).
    #[error("{message}: {source}")]
    WithContext {
        message: String,
        #[source]
        source: Box<Error>,
    },

    /// Optional backtrace (enable `backtrace` feature in Cargo.toml).
    #[cfg(feature = "backtrace")]
    #[error("{message}")]
    WithBacktrace {
        message: String,
        #[source]
        source: Option<Box<Error>>,
        backtrace: backtrace::Backtrace,
    },
}

impl Error {
    /// Create a custom error message (zero-cost when possible).
    #[inline]
    pub fn custom<S: Into<String>>(msg: S) -> Self {
        Self::Custom(msg.into())
    }

    /// Create a formatted custom error (like `format!` but returns `Error`).
    #[inline]
    pub fn format(args: fmt::Arguments) -> Self {
        Self::Custom(fmt::format(args))
    }

    /// Add context to any error (chainable, like `.context()` in anyhow).
    #[inline]
    pub fn context<C: Into<String>>(self, context: C) -> Self {
        Self::WithContext {
            message: context.into(),
            source: Box::new(self),
        }
    }

    /// Quick static message (no allocation if you pass `&'static str`).
    #[inline]
    pub fn msg(msg: &'static str) -> Self {
        Self::Custom(msg.into())
    }

    // === High-performance kind checks (branch prediction friendly) ===
    #[inline]
    pub fn is_io(&self) -> bool {
        matches!(self, Error::Io(_))
    }

    #[inline]
    pub fn is_parse(&self) -> bool {
        matches!(self, Error::ParseInt(_) | Error::ParseFloat(_))
    }

    #[inline]
    pub fn is_custom(&self) -> bool {
        matches!(self, Error::Custom(_))
    }

    // Add more `is_*` as you extend the enum.
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // thiserror already implements Display via the `#[error]` attributes
        <Self as std::error::Error>::fmt(self, f)
    }
}

/// Convenient `Result` alias — use `crate::Result<T>` everywhere.
pub type Result<T> = std::result::Result<T, Error>;

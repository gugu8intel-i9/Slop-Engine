// src/context.rs
//! High-performance context extension traits + macros for error handling.
//!
//! - **Performance**: `#[inline(always)]`, lazy `with_context` (only allocates/evaluates on error), zero overhead on `Ok` path.
//! - **Features**: anyhow/eyre-like API, `Option` support, rich macros (`bail!`, `ensure!`, `ensure_eq!`), full compatibility with `?`, tracing, async, threads.
//! - **Zero extra deps** — uses only your existing `Error` type.

use crate::error::{Error, Result};

/// Extension trait giving you `.context()` / `.with_context()` on any `Result`.
///
/// Works with foreign errors too (as long as they convert into our `Error`).
pub trait Context<T, E> {
    /// Add static or owned context (eager — use only when cheap).
    #[inline(always)]
    fn context<C>(self, context: C) -> Result<T>
    where
        C: Into<String>;

    /// Add context lazily (preferred — closure only runs on error path).
    #[inline(always)]
    fn with_context<C, F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> C,
        C: Into<String>;
}

impl<T, E> Context<T, E> for std::result::Result<T, E>
where
    E: Into<Error> + Send + Sync + 'static,
{
    #[inline(always)]
    fn context<C>(self, context: C) -> Result<T>
    where
        C: Into<String>,
    {
        match self {
            Ok(value) => Ok(value),
            Err(err) => Err(err.into().context(context)),
        }
    }

    #[inline(always)]
    fn with_context<C, F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> C,
        C: Into<String>,
    {
        match self {
            Ok(value) => Ok(value),
            Err(err) => Err(err.into().context(f())),
        }
    }
}

/// Extension trait for `Option<T>` → `Result<T, Error>` with context.
pub trait OptionContext<T> {
    #[inline(always)]
    fn context<C>(self, context: C) -> Result<T>
    where
        C: Into<String>;

    #[inline(always)]
    fn with_context<C, F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> C,
        C: Into<String>;
}

impl<T> OptionContext<T> for Option<T> {
    #[inline(always)]
    fn context<C>(self, context: C) -> Result<T>
    where
        C: Into<String>,
    {
        self.ok_or_else(|| Error::custom(context))
    }

    #[inline(always)]
    fn with_context<C, F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> C,
        C: Into<String>,
    {
        self.ok_or_else(|| Error::custom(f()))
    }
}

// ====================== CONVENIENCE MACROS ======================

/// Early return with an error — `bail!("msg")` or `bail!(err)` or formatted.
#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::error::Error::msg($msg))
    };
    ($err:expr $(,)?) => {
        return Err(Into::<$crate::error::Error>::into($err))
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::error::Error::format(format_args!($fmt, $($arg)*)))
    };
}

/// Ensure a condition is true, else `bail!`.
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $msg:literal $(,)?) => {
        if !($cond) {
            bail!($msg);
        }
    };
    ($cond:expr, $fmt:expr, $($arg:tt)*) => {
        if !($cond) {
            bail!($fmt, $($arg)*);
        }
    };
}

/// Assert two values are equal, with beautiful debug output on failure.
#[macro_export]
macro_rules! ensure_eq {
    ($left:expr, $right:expr $(,)?) => {{
        match (&$left, &$right) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    bail!(
                        "assertion failed: `({} == {})`\n  left: `{:?}`\n right: `{:?}`",
                        stringify!($left),
                        stringify!($right),
                        left_val,
                        right_val
                    );
                }
            }
        }
    }};
    ($left:expr, $right:expr, $($arg:tt)+) => {
        if !($left == $right) {
            bail!($($arg)+);
        }
    };
}

// Re-export everything for convenient `use crate::context::*;`
pub use {bail, ensure, ensure_eq, Context, OptionContext};

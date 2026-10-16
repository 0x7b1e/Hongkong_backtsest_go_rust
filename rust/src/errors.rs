//! Error kinds distinguished by the command-line interface.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    /// An output that must be new already exists (Python ``FileExistsError``).
    #[error("{0}")]
    FileExists(String),
    /// A required input is missing (Python ``FileNotFoundError``).
    #[error("{0}")]
    FileNotFound(String),
    /// A memory budget would be exceeded (Python ``MemoryError``).
    #[error("{0}")]
    Memory(String),
}

//! Streaming Hong Kong index enhancement backtesting and factor diagnostics.
//!
//! A Rust port of the ``hkfactor`` Python package with identical semantics.

#![allow(
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::excessive_precision
)]

pub mod allocation;
pub mod analysis;
pub mod config;
pub mod data;
pub mod errors;
pub mod execution;
pub mod frames;
pub mod matrix;
pub mod models;
pub mod npy;
pub mod rules;
pub mod signals;
pub mod table;
pub mod timeutil;

pub use config::BacktestConfig;
pub mod batch;
pub mod csv_data;
pub mod demo;
pub mod nprandom;
pub mod pyjson;
pub mod reporting;
pub mod signal_adapter;
pub mod tools;
pub mod validation;
pub mod viridis;
pub mod ziggurat;

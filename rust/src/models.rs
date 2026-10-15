//! Shared data contracts without execution or analysis dependencies.

use std::path::Path;

use anyhow::Result;
use serde_json::{Map, Value};

use crate::frames::{AuditTable, OutFrame};
use crate::matrix::Mat;

/// Read-only shared market arrays, shaped ``(bars, symbols)``.
///
/// Timestamps label bar ends in Asia/Hong_Kong (stored as UTC ns). Open
/// timestamps indicate executable bar starts. Base weights are point-in-time
/// snapshots known at the corresponding open. Session IDs count actual
/// supplied trading dates. Missing quotes may be carried for valuation only;
/// ``tradable`` is false for such observations. VCM bands are NaN outside an
/// explicitly supplied cooling-off interval. Cash dividends and share
/// multipliers are applied before trading at each open.
#[derive(Debug, Clone)]
pub struct MarketPanel {
    pub timestamps: Vec<i64>,
    pub open_timestamps: Vec<i64>,
    pub symbols: Vec<String>,
    pub frequency: String,
    pub open: Mat<f64>,
    pub high: Mat<f64>,
    pub low: Mat<f64>,
    pub close: Mat<f64>,
    pub volume: Mat<f64>,
    pub base_weights: Mat<f64>,
    pub benchmark_close: Vec<f64>,
    pub lot_sizes: Mat<f64>,
    pub tradable: Mat<bool>,
    pub vcm_lower: Mat<f64>,
    pub vcm_upper: Mat<f64>,
    pub vcm_eligible: Mat<bool>,
    pub cash_dividend: Mat<f64>,
    pub share_multiplier: Mat<f64>,
    pub session_id: Vec<i64>,
    pub is_session_start: Vec<bool>,
    pub is_session_end: Vec<bool>,
    pub price_basis: String,
    pub vcm_active: Option<Mat<i8>>,
    pub stamp_duty_exempt: Option<Mat<bool>>,
    pub benchmark_open: Option<Vec<f64>>,
}

impl MarketPanel {
    /// Return the bar and security dimensions.
    pub fn shape(&self) -> (usize, usize) {
        self.open.shape()
    }

    /// Persist an immutable panel for shared worker access.
    pub fn save(&self, directory: &Path) -> Result<()> {
        crate::data::save_panel(self, directory)
    }

    /// Load shared arrays.
    pub fn load(directory: &Path) -> Result<MarketPanel> {
        crate::data::load_panel(directory)
    }
}

/// One factor's values known as of each bar end.
///
/// Values are forward-filled only after ``available_at``, never backward.
/// ``updated`` flags bars receiving at least one new available observation.
/// The engine uses an earlier completed bar, never its execution bar.
#[derive(Debug, Clone)]
pub struct SignalMatrix {
    pub factor: String,
    pub values: Mat<f64>,
    pub updated: Vec<bool>,
}

/// Account results, filled orders, restrictions, and scalar metrics.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub pnl: OutFrame,
    pub trades: AuditTable,
    pub restrictions: AuditTable,
    pub metrics: Map<String, Value>,
    pub audit_streamed: bool,
}

/// Gross research layers and cross-sectional factor statistics.
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub layers: OutFrame,
    pub ic: OutFrame,
    pub ic_intraday: OutFrame,
    pub layer_summary: OutFrame,
    pub layer_pnl: Option<OutFrame>,
    pub assumptions: Map<String, Value>,
}

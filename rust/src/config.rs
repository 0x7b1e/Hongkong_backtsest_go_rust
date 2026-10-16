//! Validated, explicit research and execution assumptions.

use anyhow::{anyhow, bail, Result};
use serde_json::{Map, Number, Value};

use crate::timeutil;

/// Strategy holding convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    T0,
    T1,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::T0 => "T0",
            Mode::T1 => "T1",
        }
    }
}

/// Configure a single independent factor experiment.
///
/// T0 executes the preceding bar's known signal and restores base targets
/// from ``t0_flatten_at``, with a final-bar attempt. T1 uses the prior
/// session's known signal at the next session's open and prohibits selling
/// shares younger than ``holding_days`` trading sessions. These are strategy
/// definitions, not Hong Kong settlement restrictions.
#[derive(Debug, Clone, PartialEq)]
pub struct BacktestConfig {
    pub t_start: String,
    pub t_end: String,
    pub mode: Mode,
    pub frequency: String,
    pub keep: f64,
    pub initial_capital: f64,
    pub direction: i64,
    pub holding_days: i64,
    pub commission_bps: f64,
    pub minimum_commission: f64,
    pub slippage_bps: f64,
    pub participation_rate: f64,
    pub round_lots: bool,
    pub ic_horizon_bars: i64,
    pub ic_min_assets: i64,
    pub quantiles: i64,
    pub enforce_vcm: bool,
    pub allow_adjusted_prices: bool,
    pub entry_start: String,
    pub entry_end: String,
    pub exit_start: String,
    pub exit_end: String,
    pub t0_flatten_at: String,
    pub order_slices: i64,
    pub simulate_layers: bool,
    /// Original JSON numbers, preserved for ``to_dict`` round trips.
    raw_numbers: Map<String, Value>,
}

const FIELDS: &[&str] = &[
    "t_start",
    "t_end",
    "mode",
    "frequency",
    "keep",
    "initial_capital",
    "direction",
    "holding_days",
    "commission_bps",
    "minimum_commission",
    "slippage_bps",
    "participation_rate",
    "round_lots",
    "ic_horizon_bars",
    "ic_min_assets",
    "quantiles",
    "enforce_vcm",
    "allow_adjusted_prices",
    "entry_start",
    "entry_end",
    "exit_start",
    "exit_end",
    "t0_flatten_at",
    "order_slices",
    "simulate_layers",
];

impl BacktestConfig {
    /// Construct with defaults, as ``BacktestConfig(t_start, t_end)``.
    pub fn new(t_start: &str, t_end: &str) -> Result<Self> {
        let mut values = Map::new();
        values.insert("t_start".into(), Value::String(t_start.into()));
        values.insert("t_end".into(), Value::String(t_end.into()));
        Self::from_map(&values)
    }

    /// Construct from keyword values; unknown keys are rejected like
    /// ``BacktestConfig(**values)``.
    pub fn from_map(values: &Map<String, Value>) -> Result<Self> {
        for key in values.keys() {
            if !FIELDS.contains(&key.as_str()) {
                bail!("BacktestConfig.__init__() got an unexpected keyword argument '{key}'");
            }
        }
        for required in ["t_start", "t_end"] {
            if !values.contains_key(required) {
                bail!("BacktestConfig.__init__() missing required argument: '{required}'");
            }
        }
        let get = |name: &str| values.get(name);
        let text = |name: &str, default: &str| -> Result<String> {
            match get(name) {
                None => Ok(default.to_string()),
                Some(Value::String(s)) => Ok(s.clone()),
                Some(other) => bail!("{name} must be a string, got {other}"),
            }
        };
        let float = |name: &str, default: f64| -> Result<f64> {
            match get(name) {
                None => Ok(default),
                Some(Value::Number(n)) => Ok(n.as_f64().unwrap()),
                Some(Value::Bool(b)) => Ok(*b as i64 as f64),
                Some(other) => bail!("{name} must be numeric, got {other}"),
            }
        };
        // Python requires ``type(value) is int`` for these fields.
        let integer = |name: &str, default: i64| -> Result<Option<i64>> {
            match get(name) {
                None => Ok(Some(default)),
                Some(Value::Number(n)) if n.is_i64() => Ok(Some(n.as_i64().unwrap())),
                Some(_) => Ok(None),
            }
        };
        let toggle = |name: &str, default: bool| -> Result<Option<bool>> {
            match get(name) {
                None => Ok(Some(default)),
                Some(Value::Bool(b)) => Ok(Some(*b)),
                Some(_) => Ok(None),
            }
        };
        // Date bounds must be ISO strings.
        for name in ["t_start", "t_end"] {
            match get(name) {
                Some(Value::String(s)) => {
                    timeutil::parse_iso_date(s)?;
                }
                _ => bail!("Date bounds must be ISO YYYY-MM-DD strings"),
            }
        }
        let integers = [
            integer("holding_days", 1)?,
            integer("ic_horizon_bars", 1)?,
            integer("ic_min_assets", 3)?,
            integer("quantiles", 10)?,
            integer("order_slices", 1)?,
            integer("direction", 1)?,
        ];
        if integers.iter().any(Option::is_none) {
            bail!("Periods, layers, slices and direction need ints");
        }
        let toggles = [
            toggle("round_lots", true)?,
            toggle("enforce_vcm", false)?,
            toggle("allow_adjusted_prices", false)?,
            toggle("simulate_layers", true)?,
        ];
        if toggles.iter().any(Option::is_none) {
            bail!("Execution toggles must be true or false");
        }
        let mode_text = text("mode", "T1")?;
        let config = BacktestConfig {
            t_start: text("t_start", "")?,
            t_end: text("t_end", "")?,
            mode: match mode_text.as_str() {
                "T0" => Mode::T0,
                "T1" => Mode::T1,
                _ => Mode::T1,
            },
            frequency: text("frequency", "1d")?,
            keep: float("keep", 0.9)?,
            initial_capital: float("initial_capital", 10_000_000.0)?,
            direction: integers[5].unwrap(),
            holding_days: integers[0].unwrap(),
            commission_bps: float("commission_bps", 2.0)?,
            minimum_commission: float("minimum_commission", 0.0)?,
            slippage_bps: float("slippage_bps", 0.0)?,
            participation_rate: float("participation_rate", 0.1)?,
            round_lots: toggles[0].unwrap(),
            ic_horizon_bars: integers[1].unwrap(),
            ic_min_assets: integers[2].unwrap(),
            quantiles: integers[3].unwrap(),
            enforce_vcm: toggles[1].unwrap(),
            allow_adjusted_prices: toggles[2].unwrap(),
            entry_start: text("entry_start", "09:30")?,
            entry_end: text("entry_end", "16:00")?,
            exit_start: text("exit_start", "09:30")?,
            exit_end: text("exit_end", "16:00")?,
            t0_flatten_at: text("t0_flatten_at", "15:30")?,
            order_slices: integers[4].unwrap(),
            simulate_layers: toggles[3].unwrap(),
            raw_numbers: values
                .iter()
                .filter(|(_, v)| v.is_number())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        };
        let start = timeutil::parse_iso_date(&config.t_start)?;
        let end = timeutil::parse_iso_date(&config.t_end)?;
        if start > end {
            bail!("t_start must be on or before t_end");
        }
        if mode_text != "T0" && mode_text != "T1" {
            bail!("mode must be T0 or T1");
        }
        if !["1d", "5m", "1m"].contains(&config.frequency.as_str()) {
            bail!("frequency must be 1d, 5m, or 1m");
        }
        if config.mode == Mode::T0 && config.frequency == "1d" {
            bail!("T0 requires individual-stock intraday bars");
        }
        if !(0.0..=1.0).contains(&config.keep) {
            bail!("keep must be between 0 and 1");
        }
        if config.initial_capital <= 0.0 || ![-1, 1].contains(&config.direction) {
            bail!("Invalid capital or factor direction");
        }
        let finite = [
            config.initial_capital,
            config.keep,
            config.commission_bps,
            config.minimum_commission,
            config.slippage_bps,
            config.participation_rate,
        ];
        if !finite.iter().all(|v| v.is_finite()) {
            bail!("Numeric configuration must be finite");
        }
        if config.holding_days < 1 || config.ic_horizon_bars < 1 {
            bail!("Holding periods must be positive");
        }
        if config.ic_min_assets < 3 || config.quantiles < 2 {
            bail!("Invalid IC sample size or quantile count");
        }
        if !(config.participation_rate > 0.0 && config.participation_rate <= 1.0) {
            bail!("participation_rate must be in (0, 1]");
        }
        if config
            .commission_bps
            .min(config.minimum_commission)
            .min(config.slippage_bps)
            < 0.0
        {
            bail!("Transaction costs cannot be negative");
        }
        if config.order_slices < 1 {
            bail!("order_slices must be positive");
        }
        let entry = (
            timeutil::parse_iso_time(&config.entry_start)?,
            timeutil::parse_iso_time(&config.entry_end)?,
        );
        let exit = (
            timeutil::parse_iso_time(&config.exit_start)?,
            timeutil::parse_iso_time(&config.exit_end)?,
        );
        timeutil::parse_iso_time(&config.t0_flatten_at)?;
        if entry.0 >= entry.1 || exit.0 >= exit.1 {
            bail!("Execution-window start must precede end");
        }
        Ok(config)
    }

    /// Parse a JSON object of configuration values.
    pub fn from_json(value: &Value) -> Result<Self> {
        let map = value
            .as_object()
            .ok_or_else(|| anyhow!("Configuration must be a JSON object"))?;
        Self::from_map(map)
    }

    /// Copy with ``simulate_layers`` replaced, like ``dataclasses.replace``.
    pub fn with_simulate_layers(&self, value: bool) -> Self {
        let mut copy = self.clone();
        copy.simulate_layers = value;
        copy
    }

    /// Copy with ``keep`` replaced.
    pub fn with_keep(&self, value: f64) -> Result<Self> {
        let mut map = self.to_dict();
        map.insert("keep".into(), json_float(value));
        Self::from_map(&map)
    }

    fn number(&self, name: &str, value: f64) -> Value {
        match self.raw_numbers.get(name) {
            Some(Value::Number(n)) if n.as_f64() == Some(value) => Value::Number(n.clone()),
            _ => json_float(value),
        }
    }

    /// Return JSON-serializable configuration values in field order.
    pub fn to_dict(&self) -> Map<String, Value> {
        let mut map = Map::new();
        map.insert("t_start".into(), Value::String(self.t_start.clone()));
        map.insert("t_end".into(), Value::String(self.t_end.clone()));
        map.insert("mode".into(), Value::String(self.mode.as_str().into()));
        map.insert("frequency".into(), Value::String(self.frequency.clone()));
        map.insert("keep".into(), self.number("keep", self.keep));
        map.insert(
            "initial_capital".into(),
            self.number("initial_capital", self.initial_capital),
        );
        map.insert("direction".into(), Value::from(self.direction));
        map.insert("holding_days".into(), Value::from(self.holding_days));
        map.insert(
            "commission_bps".into(),
            self.number("commission_bps", self.commission_bps),
        );
        map.insert(
            "minimum_commission".into(),
            self.number("minimum_commission", self.minimum_commission),
        );
        map.insert(
            "slippage_bps".into(),
            self.number("slippage_bps", self.slippage_bps),
        );
        map.insert(
            "participation_rate".into(),
            self.number("participation_rate", self.participation_rate),
        );
        map.insert("round_lots".into(), Value::Bool(self.round_lots));
        map.insert("ic_horizon_bars".into(), Value::from(self.ic_horizon_bars));
        map.insert("ic_min_assets".into(), Value::from(self.ic_min_assets));
        map.insert("quantiles".into(), Value::from(self.quantiles));
        map.insert("enforce_vcm".into(), Value::Bool(self.enforce_vcm));
        map.insert(
            "allow_adjusted_prices".into(),
            Value::Bool(self.allow_adjusted_prices),
        );
        map.insert(
            "entry_start".into(),
            Value::String(self.entry_start.clone()),
        );
        map.insert("entry_end".into(), Value::String(self.entry_end.clone()));
        map.insert("exit_start".into(), Value::String(self.exit_start.clone()));
        map.insert("exit_end".into(), Value::String(self.exit_end.clone()));
        map.insert(
            "t0_flatten_at".into(),
            Value::String(self.t0_flatten_at.clone()),
        );
        map.insert("order_slices".into(), Value::from(self.order_slices));
        map.insert("simulate_layers".into(), Value::Bool(self.simulate_layers));
        map
    }

    pub fn entry_window(&self) -> (i64, i64) {
        (
            timeutil::parse_iso_time(&self.entry_start).unwrap(),
            timeutil::parse_iso_time(&self.entry_end).unwrap(),
        )
    }

    pub fn exit_window(&self) -> (i64, i64) {
        (
            timeutil::parse_iso_time(&self.exit_start).unwrap(),
            timeutil::parse_iso_time(&self.exit_end).unwrap(),
        )
    }

    pub fn flatten_time(&self) -> i64 {
        timeutil::parse_iso_time(&self.t0_flatten_at).unwrap()
    }
}

/// A JSON float that keeps a fractional representation for whole values.
pub fn json_float(value: f64) -> Value {
    Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_and_validation() {
        let config = BacktestConfig::new("2024-01-02", "2024-01-03").unwrap();
        assert_eq!(config.mode, Mode::T1);
        assert_eq!(config.keep, 0.9);
        let bad = json!({"t_start": "2024-01-02", "t_end": "2024-01-03", "mode": "T0"});
        assert!(BacktestConfig::from_json(&bad).is_err());
        let bad = json!({"t_start": "2024-01-02", "t_end": "2024-01-03", "quantiles": 10.0});
        assert!(BacktestConfig::from_json(&bad).is_err());
        let bad = json!({"t_start": "2024-01-02", "t_end": "2024-01-03", "unknown": 1});
        assert!(BacktestConfig::from_json(&bad).is_err());
    }
}

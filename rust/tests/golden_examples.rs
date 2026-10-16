//! Regression tests against results produced by the Python implementation.
//!
//! ``tests/golden/examples.json`` holds the Python metrics, IC series, layer
//! summaries and NAV curves for both bundled examples. These tests rebuild
//! each panel from the raw example inputs and require the Rust engine to
//! reproduce them to a relative tolerance of 1e-9.

use std::path::{Path, PathBuf};

use hkfactor::analysis;
use hkfactor::config::BacktestConfig;
use hkfactor::data;
use hkfactor::execution;
use hkfactor::signals::{SignalColumns, SignalStore};
use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn close(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 + 1e-9 * b.abs()
}

fn assert_series(name: &str, actual: &[f64], expected: &Value) {
    let expected: Vec<f64> = expected
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect();
    assert_eq!(actual.len(), expected.len(), "{name}: length");
    for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
        assert!(close(*a, *b), "{name}[{i}]: {a} != {b}");
    }
}

fn check(case: &str) {
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(root().join("rust/tests/golden/examples.json")).unwrap(),
    )
    .unwrap();
    let source = root().join("examples").join(case);
    let config_value: Value =
        serde_json::from_str(&std::fs::read_to_string(source.join("config.json")).unwrap())
            .unwrap();
    let config = BacktestConfig::from_json(&config_value).unwrap();
    let scratch = tempfile::tempdir().unwrap();
    let panel = data::prepare_market(
        &source.join("prices.parquet"),
        &source.join("benchmark.parquet"),
        &source.join("constituents.parquet"),
        &scratch.path().join("market"),
        &config.frequency,
        Some(&source.join("sessions.parquet")),
        Some(&config.t_start),
        Some(&config.t_end),
        1 << 32,
    )
    .unwrap();
    let store = SignalStore::new(&source.join("signals"), SignalColumns::default()).unwrap();
    for factor in store.factors().unwrap() {
        let expected = &golden[format!("{case}/{factor}")];
        let signal = store.read_factor(&factor, &panel, None).unwrap();
        let result = execution::run_backtest(&panel, &signal, &config, None).unwrap();
        for (key, value) in expected["metrics"].as_object().unwrap() {
            let actual = &result.metrics[key];
            match value {
                Value::Number(n) if n.is_f64() => assert!(
                    close(actual.as_f64().unwrap(), n.as_f64().unwrap()),
                    "{case}/{factor} {key}: {actual} != {value}"
                ),
                other => assert_eq!(actual, other, "{case}/{factor} {key}"),
            }
        }
        assert_series(
            &format!("{case}/{factor} net_nav"),
            result.pnl.col("net_nav").f64s(),
            &expected["net_nav"],
        );
        let research = analysis::analyze_factor(&panel, &signal, &config).unwrap();
        let ic: Vec<f64> = research
            .ic
            .col("pearson_ic")
            .f64s()
            .iter()
            .map(|v| if v.is_nan() { -9.0 } else { *v })
            .collect();
        assert_series(&format!("{case}/{factor} ic"), &ic, &expected["ic_pearson"]);
        assert_series(
            &format!("{case}/{factor} layers"),
            research.layer_summary.col("mean_one_bar_return").f64s(),
            &expected["layer_mean_one_bar"],
        );
    }
}

#[test]
fn daily_example_matches_python() {
    check("daily");
}

#[test]
fn intraday_example_matches_python() {
    check("intraday");
}

# hkfactor（Rust 版）

`src/hkfactor` Python 包的 Rust 全量移植，回测逻辑与 Python 版完全一致：同样的输入产生同样的交易、限制记录、净值、IC 和分层结果。Python 版中已知的逻辑问题（日频 IC 只算开盘到收盘、分层账户使用 `keep` 等）**按原样保留**，没有修改。

## 构建

```bash
cd rust
cargo build --release          # 产物：rust/target/release/hkfactor
cargo test --release           # 单元测试 + 以 Python 结果为基准的回归测试
```

需要 Rust 1.80 及以上版本。

## 命令（与 `python -m hkfactor` 一一对应）

```bash
hkfactor demo     --output demo_daily --frequency 1d
hkfactor prepare  --prices P --benchmark B --constituents C --sessions S \
                  --frequency 1d --t-start 2024-01-02 --t-end 2024-02-26 --output market
hkfactor run      --market market --signals signals --config config.json \
                  --output results [--workers 4] [--factor momentum_5] \
                  [--no-resume] [--no-plots] [--no-layer-execution]
hkfactor audit    --data data --output audit.json
hkfactor import-csv     --input x.csv --output x.parquet --symbol 00005 --frequency 1d --kind stock [--price-basis raw]
hkfactor import-signals --input wide/ --output signals/ --timestamp-label bar_end [--timestamp-unit ms] ...
hkfactor compare  --local a.parquet --reference b.parquet --output cmp/
```

另外，`scripts/` 下的两个脚本也移植成了子命令：

```bash
hkfactor benchmark   --output bench/ [--factors 16 --days 1008 --assets 80 --workers 2 --layers]
hkfactor plot-readme --t0 results_t0/ --t1 results_t1/ --output docs/images
```

参数、配置文件（JSON/TOML，含 `backtest` 段和 `signal_columns` 映射）、输出目录结构、文件名、Parquet 列与 JSON 字段都与 Python 版相同。panel 缓存仍然是 `.npy` + `metadata.json` 格式，两种实现可以互相读取对方生成的缓存。

## 模块对照

| Python | Rust |
|---|---|
| `config.py` | `config.rs` |
| `rules.py` | `rules.rs` |
| `allocation.py` | `allocation.rs` |
| `models.py` | `models.rs`、`frames.rs` |
| `data.py` | `data.rs`、`npy.rs`、`table.rs`、`timeutil.rs` |
| `signals.py` | `signals.rs` |
| `execution.py` | `execution.rs` |
| `analysis.py` | `analysis.rs` |
| `audit.py` | `frames.rs`（`AuditRecords`） |
| `reporting.py` | `reporting.rs`（matplotlib 换成 plotters） |
| `batch.py` | `batch.rs`（多进程换成共享 panel 的多线程） |
| `cli.py` | `main.rs` |
| `csv_data.py` | `csv_data.rs` |
| `validation.py` | `validation.rs` |
| `signal_adapter.py` | `signal_adapter.rs` |
| `demo.py` | `demo.rs`、`nprandom.rs`（逐位复刻 `np.random.default_rng`） |
| `scripts/benchmark.py`、`scripts/plot_readme.py` | `tools.rs` |

## 一致性验证

`parity/compare.py` 会分别用两种实现执行 `prepare` 和 `run`，然后比较 panel 的每个数组，以及每个因子、每个分层账户输出的所有 Parquet 和 JSON：

```bash
python parity/make_stress.py      --output /tmp/stress       # 分红、拆股、停牌、缺报价、VCM、印花税豁免、换仓、延迟/过期/清空信号、date 类型信号
python parity/make_panel_case.py  --output /tmp/panel_case   # 只有手工构造 panel 才能触发的执行分支
python parity/compare.py --rust rust/target/release/hkfactor \
  --case examples/daily --case examples/intraday \
  --case /tmp/stress/stress_daily --case /tmp/stress/stress_intraday \
  --case /tmp/panel_case --work /tmp/parity
```

判定标准：
- panel 数组必须逐位相等。
- 行数、时间戳、代码、方向、原因等非浮点字段必须完全相同。
- 浮点数的相对误差不超过 1e-9。只有两类字段放宽绝对容差：PnL（约 1e7 港元净值之差）放宽到 1e-6 港元；股数类字段（例如保底缺口）放宽到 1e-6 股。这两类字段都是大数相减的结果，NumPy 的 BLAS 点积与 Rust 的累加顺序不同，会在最后一位产生差异，相减后被放大。

## 已知差异

- **demo 合成数据**：随机数抽样逐位一致。但在支持 AVX512 的 CPU 上，NumPy 的 `exp` 走 SIMD 实现，约 4% 的结果与标准数学库差最后一位，所以 `demo --frequency 1m` 的价格可能有末位差异（1d、5m 在本机测试完全一致）。Python 自身在不同 CPU 上的结果也会因此不同。
- **Parquet 文件字节**：编码器不同，文件大小（例如 `import_manifest.json` 中的 `output_size_bytes`）会有差别，读出的数据完全相同。
- **图表**：使用 plotters 渲染，布局与 matplotlib 相近，但不是像素级一致。
- **实现指纹**：批量运行的 `identity.json` 记录的是 Rust 源码的哈希和库版本，因此 Python 和 Rust 的运行结果不会互相命中断点续跑缓存。
- **`compare_with_provider`**：Python 版接收 Python 对象作为行情源；Rust 版改为实现 `validation::PriceProvider` trait。

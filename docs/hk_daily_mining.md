# 港股日频因子挖掘

本分支沿用 Efficient Cryptos 的“候选配方 → 时间审计 → 批量回测 →
训练/验证筛选 → 锁定后测试”流程。候选是受限 JSONL 配方，不运行任意 Python
代码；交易、费用、持仓、限制与账户净值统一由本仓库的 Rust 回测执行。

## 1. 数据与不能自动推断的字段

数据集：[Crypt-dataset/hongkong_day_kline](https://huggingface.co/datasets/Crypt-dataset/hongkong_day_kline)。
截至接入时，数据仓库需要授权，公开说明没有 CSV 字段定义；目录包含
`不复权/<代码>.HK_<名称>.csv` 和 `后复权/<代码>.HK_<名称>.csv`。
先获得授权并下载到本地，再检查一份真实 CSV 的表头、成交量单位、
日期和复权定义。以下命令中的 `shares` 是**经人工核实后的声明**，
并不代表本项目已证实数据集的成交量单位：

```bash
uv sync --extra dev --frozen
uv run hkfactor-mine import-hf \
  --dataset-root /path/to/hongkong_day_kline \
  --basis raw --volume-unit shares --output data/hk_raw
```

导入会逐证券验证日期、代码、OHLC、成交量、重复记录和价格口径，按证券写
Parquet，并记录源文件 SHA256 和列映射于 `data/hk_raw/_import_manifest.json`。
若真实列名不在默认别名中，用 `--columns columns.json`，例如：

```json
{"timestamp":"交易日","open":"开盘价格","high":"最高价格","low":"最低价格","close":"收盘价格","volume":"交易股数"}
```

映射键可以包括 `symbol`、`lot_size`、`cash_dividend` 和
`share_multiplier`。含糊或缺失的必需列会直接报错。可单独导入
`--basis adjusted --output data/hk_adjusted` 做价格研究；挖掘命令只接受
**不复权的原始价**执行面板。复权价不能代替真实拆股和分红现金流。

`hkfactor prepare` 仍然需要**外部提供**覆盖同一交易日的基准行情
（`timestamp,close`，最好有 `open`）和历史时点成分股快照
（`effective_date,available_at,symbol,weight`；每张完整快照权重和为 1）。
不得把今天的成分股名单回填到历史。真实按手交易还需要每只股票的历史
`lot_size`；准确的总回报还需要每股派息与股份倍数。若 HF 文件未包含这些
字段，先将有发布日期的历史来源合并进原始价 Parquet，再准备面板。

```bash
uv run hkfactor prepare \
  --prices data/hk_raw --benchmark /path/to/benchmark.parquet \
  --constituents /path/to/pit_constituents.parquet \
  --frequency 1d --output cache/hk_daily
```

若缺历史每手股数，仅可在研究配置中设置 `round_lots=false`；
若已经核对公司行动数据的完整性，运行时加 `--corporate-actions-verified`。
若缺公司行动，仅可显式加 `--price-only-research` 观察不含分红/拆股的
探索性结果。这样的结果不能解释为准确的可交易总回报。

## 2. 挖掘与筛选

```bash
cd rust && cargo build --release && cd ..
uv run hkfactor-mine recipes --output data/hk_recipes.jsonl
uv run hkfactor-mine run \
  --market cache/hk_daily --config /path/to/daily_config.json \
  --recipes data/hk_recipes.jsonl --train-end 2022-12-30 \
  --valid-end 2024-12-31 --output outputs/hk_mining_001 \
  --corporate-actions-verified
```

`daily_config.json` 使用现有 `hkfactor run` 配置，`t_start` 到 `t_end`
应覆盖训练、验证和留出的测试期；要求 `frequency=1d`、`mode=T1`、
`direction=1` 和 `allow_adjusted_prices=false`。`--rust-bin` 默认为
`rust/target/release/hkfactor`，`--workers` 控制并行。运行不能覆盖已有输出。
开发验证可用 `--engine python`，正式实验应使用 Rust 二进制。

配方 JSONL 每行例如：

```json
{"factor_id":"momentum_20","kind":"momentum","window":20,"sign":1,"hypothesis":"价格延续"}
```

允许的 `kind`：`momentum`（过去 N 日含已供应公司行动的累计收益）、
`volatility`（N 日含已供应公司行动的收益标准差）、`relative_volume`
（当日成交股数相对于前 N 日均值）、
`intraday`（当日开收盘涨幅）、`range`（N 日平均振幅）。`sign=-1`
令低原始值对应高因子得分。默认含 10 个候选；可以人工或模型生成新 JSONL
候选，但只能采用这些可审计模板。每只股票只读取当前及以前的 K 线，
并抽样执行前缀不变性审计；零成交日信号清空。
用模型提出新候选时可采用[港股日频候选提示词](hk_daily_factor_prompt.md)，
再人工检查假设及 JSONL。

每个候选以 `timestamp,symbol,value,available_at` 写入
`signals/factor=<factor_id>/...parquet`；Rust `SignalStore` 会做按发布时间对齐。
默认 `available_at` 是日线标记时间后 10 分钟，因此按照目前回测器的
“上一根已完成 bar 信号，下根开盘下单”接口，**在 D 日收盘形成的候选要到
D+2 个交易日开盘才会执行**。仅在有证据证明最终日线在港时 16:00 已经
可用时，才用 `--publication-delay 0min` 获得 D+1 开盘的假设。香港收市
竞价及供应商发布时间应结合实盘数据核对；`0min` 不是默认值。

训练按 Rust 输出的**扣费净超额累计收益**排名，用训练期截面信号相关系数
去重，保留前 `--top-k`。验证期只评价这些候选，默认要求净超额收益非负，
再保留前 `--select-k`。`selection.json` 在测试运行前写入；测试期不参与
筛选。`ic.parquet` 汇总为报告诊断，但当前引擎的日频 IC 是**次日开盘到
收盘**，不能当作隔夜/收盘到收盘标签。分段回测各自从初始资金开始，
研究期指标不可当成单条连续实盘净值。

结果位于 `outputs/hk_mining_001/manifest.json`、`selection.json`、
`report.json` 及 `train/`、`validation/`、`test/` 的原生 Rust 报告。
报告保留输入指纹、引擎二进制 SHA256、配方和发布日期假设。

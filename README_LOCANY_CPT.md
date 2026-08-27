# LocateAnything UI CPT v2

本目录只说明 CPT。CPT 运行必须设置 `LOCANY_CPT_MODE=1`；新增统计、采样校验和
UI5 检查旁路均受此开关保护，不改变 SFT 默认行为。

UI CPT v2 的目标是把训练改造成可观测、可诊断、可比较的系统：按图片组切分
held-out、精确核对样本与监督 token、按任务计算 train/eval CE、用任务专属指标评估，
并让 JSON/JSONL 成为唯一数据源。Excel 只是可选离线投影。

## 1. 数据准备与无泄漏切分

默认按图片内容 SHA-256 生成 `group_id`，固定 seed `20260826`，以 group 为单位近似
98%/2% 切分。同一图片派生的不同任务和 annotation row 只能进入同一 split；若多图记录
与其他记录共享任意一张图片，会按共享图片的连通分量合并为同一 group，避免局部重叠泄漏。
只有在全量内容哈希确实不可承受时才使用 `--group-id-mode path`；该模式会额外输出
`diagnostics/path_duplicate_suspects.json`，列出同 basename、同尺寸但路径不同的候选，
必须抽样复核，不能将它当成与内容哈希同等强度的零泄漏证据。

```bash
python scripts/prepare_locany_cpt.py \
  --source-root /path/to/raw_data_v4.1_hl \
  --output-dir /path/to/locany_cpt_v4_split_v2 \
  --recipe-name locany_cpt_train.json \
  --split-seed 20260826 \
  --val-fraction 0.02 \
  --val-fast-per-task 200 \
  --group-id-mode sha256 \
  --overwrite
```

集群上不要复用上一轮的 `locany_cpt_v4`。仓库提供了可直接生成并完整校验
train/val/val_fast、manifest 的命令；smoke 默认每任务先取 2000 条、val_fast 至少 10 条：

```bash
bash shell/prepare_locany_cpt_v2.sh h20 smoke
bash shell/prepare_locany_cpt_v2.sh h20 formal
```

已有同名 v2 目录时命令会拒绝覆盖；只有明确要重建时才设置 `OVERWRITE=1`。A800
文件系统使用 `a100` 参数；若原始数据不在默认位置，显式设置 `SOURCE_ROOT`。

输出包括：

- `train/<task>.jsonl`、`val/<task>.jsonl`；
- `recipe/locany_cpt_train.json`、`locany_cpt_val.json`、
  `locany_cpt_val_fast.json`；
- `diagnostics/split_manifest.jsonl`、`split_summary.json`、图片哈希缓存；
- 被格式转换拒绝的行及真实原因。

若已有规范化的未切分 recipe，可单独切分：

```bash
python scripts/split_locany_cpt.py \
  --recipe /path/to/locany_cpt_all.json \
  --output-dir /path/to/locany_cpt_v4 \
  --seed 20260826 \
  --val-fraction 0.02 \
  --val-fast-per-task 200
```

全量验证必须读取 manifest；发现 group、record、图片内容哈希或规范化路径跨 split 时
非零退出：

```bash
python scripts/validate_locany_cpt.py \
  --recipe /path/to/locany_cpt_v4/recipe/locany_cpt_train.json \
  --records-per-dataset 0 \
  --split-manifest /path/to/locany_cpt_v4/diagnostics/split_manifest.jsonl \
  --require-split train \
  --require-equal-weights

python scripts/validate_locany_cpt.py \
  --recipe /path/to/locany_cpt_v4/recipe/locany_cpt_val.json \
  --records-per-dataset 0 \
  --split-manifest /path/to/locany_cpt_v4/diagnostics/split_manifest.jsonl \
  --require-split heldout
```

`val_fast` 不是取文件前 N 条，而是按 `seed + group_id + record_id` 的 hash 固定选择；
VQA 和 UI defect 保持标签分层。

## 2. 静态长度与采样模拟

在训练前用真实 processor 统计 pre/post-MTP 长度、main/MTP token 及超限原因，并把
每任务平均监督 token 写回 recipe，供 token-aware 策略使用：

```bash
python scripts/analyze_locany_cpt_lengths.py \
  --recipe /path/to/locany_cpt_v4/recipe/locany_cpt_train.json \
  --processor /path/to/LocateAnything-3B \
  --output /path/to/locany_cpt_v4/diagnostics/cpt_data_stats.json \
  --block-size 6 \
  --max-num-tokens-per-sample 8192
```

静态分析同时写 `diagnostics/oversize_samples.jsonl`，逐条保存 task、record/group、
source/line、pre/post-MTP 长度和超限原因。训练 JSONL 的
`window_oversize_record_hashes`/`window_oversize_group_hashes` 记录该统计窗口首次出现的
真实 runtime skip，可与 manifest 的稳定 ID 对照且 resume 后不会重复写。

离线比较四种固定采样；这不会启动训练：

```bash
python scripts/simulate_locany_cpt_sampling.py \
  --recipe /path/to/locany_cpt_v4/recipe/locany_cpt_train.json \
  --data-stats /path/to/locany_cpt_v4/diagnostics/cpt_data_stats.json \
  --output /path/to/locany_cpt_v4/diagnostics/cpt_sampling_simulation.json \
  --optimizer-steps 20000 \
  --samples-per-step 23.57
```

采样统一使用 `p_i ∝ N_i^alpha × mu_i^(-beta)`：

| `CPT_SAMPLING_MODE` | alpha | beta | 含义 |
| --- | ---: | ---: | --- |
| `sample_equal` | 0 | 0 | 任务样本等权，formal 默认 |
| `sqrt_size` | 0.5 | 0 | 增加大任务覆盖 |
| `token_balanced` | 0 | 1 | 近似任务监督 token 等权 |
| `hybrid` | 0.5 | 0.5 | 覆盖与 token 平衡折中 |

可覆盖 `CPT_SIZE_ALPHA`、`CPT_TOKEN_BETA`、`CPT_MIN_TASK_PROB`、
`CPT_MAX_TASK_PROB`。概率在训练前固定并写入 run config；resume 时必须一致，训练中
不会根据 rolling statistics 动态改权重。

## 3. 训练和 smoke

第一阶段 formal 保持 `CPT_SAMPLING_MODE=sample_equal`：

```bash
CPT_SAMPLING_MODE=sample_equal \
RUN_NAME=locany-3b-ui-cpt-v4-v2-h20x2-formal \
bash shell/run_locany_cpt.sh h20 formal
```

Merlin 入口：

```bash
mlx job submitv2 --path locany_cpt_v4_a100x4_smoke_merlin.yaml
mlx job submitv2 --path locany_cpt_v4_h20x2_smoke_merlin.yaml
mlx job submitv2 --path locany_cpt_v4_a100x4_formal_merlin.yaml
mlx job submitv2 --path locany_cpt_v4_h20x2_formal_merlin.yaml
```

两份 smoke 都先在 step 10 强制保存并退出 segment，再从同一 checkpoint 自动 resume 到
step 20，因此一个 job 同时覆盖多卡训练与断点续训。H20×2 默认每 rank packed-token 上限
7268、单样本与序列上限也为 7268、SDPA、packing buffer 16、梯度累积 4；
A800×4 默认 SDPA + 7268/7268/12800、梯度累积 2。`shell/run_locany_cpt.sh`
会先验证 train split，再将 split/length stats 复制到 run 的 `diagnostics/`。

H20 的 Magi 8192/7280 已实测 OOM，Magi 6400 已出现 gather index 越界，因此 v2 H20
smoke/formal 均不再把 Magi 或 25600 作为默认值。所有 v2 YAML 使用全新的
`DATA_DIR=.../locany_cpt_v4_split_v2[_smoke]` 与带 `v2` 的 `RUN_NAME`；新实验从原始
LocateAnything-3B checkpoint 开始，checkpoint-1549/1860 仅用于对照评测，禁止自动续训。

训练运行时只按区间写聚合统计，不逐样本打印：

- `CPT_METRICS_INTERVAL=100`：JSONL 和常规 logger/W&B/TensorBoard 标量；
- `CPT_TABLE_INTERVAL=500`：精确全局 unique coverage 和紧凑 per-task 表；
- stream 底层日志仍为每 10,000 packed batch。

## 4. 指标口径

样本计数满足：

```text
attempted_samples = accepted_samples + oversize_skipped_samples
```

`attempted` 是 iterator 取出的样本，`accepted` 是通过单样本 post-MTP 上限的样本，
`trained` 是实际进入 forward 的子样本，`packed_batches` 是产生的物理 packed batch。
图片读取、processor 或 token 对齐异常不会伪装成 oversize skip，而会记录 task/record/source
后抛出原始异常。

监督 token 满足：

```text
total_supervised_tokens = main_supervised_tokens + mtp_supervised_tokens
```

packed sequence 内每个 loss token 都保留 task id 和 supervision kind。统计 CE 使用同一次
forward 的 causal-shift unreduced CE，detach/no-grad 聚合，不做第二次 forward，也不改变
反向传播 loss：

- `train_main_token_ce`：main LM token CE；
- `train_mtp_token_ce`：MTP token CE；
- `train_total_token_ce`：两者按 token count 合并；
- held-out teacher-forced 使用 `eval_main_token_ce`；train–val gap 始终比较 main CE。

全局 CE 是 `sum(loss_sum) / sum(token_count)`，不是 rank mean 的均值。Unique record/group
通过可合并 64-bit hash set 全局 union，不能将各 rank unique 数直接相加。

覆盖率定义：

```text
effective_epoch = trained_samples / dataset_rows
repeat_factor   = trained_samples / max(unique_record_count, 1)
```

全局 oversize rate >0.5% 或任一任务 >2% 只报警；token share/sample share >2 标记为
`token_dominant`。不会自动扩大 token limit 或截断 ground truth。

## 5. Resume

Trainer checkpoint 保存 optimizer、scheduler、random state，以及每 rank 的
`dataloader_state_rank<N>.pt`。CPT resume 会校验：

- sampling config hash、各任务概率与 rows；
- world size、worker 数、base seed、单样本/packed token 上限、buffer size；
- metrics JSONL 不得领先所选 checkpoint；
- dataloader state 必须存在且版本兼容。

校验不通过时故意失败，避免静默从头取样造成重复计数。换 world size/worker 数或 sampling
策略应使用新的 `OUTPUT_DIR`。TorchElastic 入口带 `@record`，rank 失败保留原始异常、文件、
行号并以非零状态退出。

每个通过完整性校验的 checkpoint 还会由 rank 0 去重追加
`diagnostics/cpt_eval_queue.jsonl`，标记待独立 job 执行的 held-out val_fast 评测。完成标记
会先原子落盘、随后才发布 queue row，避免评测器抢到尚未完成的 checkpoint。queue row
具有 pending/running/completed/failed 状态；失败项只能显式设置 `EVAL_RETRY_FAILED=1` 重试。

## 6. Held-out 评测

训练池只能显式标记为 `train_pool/domain_absorption`；best checkpoint 只看 held-out。
推荐用独立 Merlin job 跑 generation，避免阻塞 formal training。

```bash
python scripts/eval_locany_cpt_learning.py \
  --checkpoint "$CKPT" \
  --base-model "$BASE" \
  --processor-path "$BASE" \
  --recipe /path/to/locany_cpt_v4/recipe/locany_cpt_val_fast.json \
  --manifest /path/to/locany_cpt_v4/diagnostics/split_manifest.jsonl \
  --eval-split heldout \
  --subset-strategy hash \
  --samples-per-task 200 \
  --base-cache-dir "$RUN_DIR/eval/base_cache" \
  --train-metrics-jsonl "$RUN_DIR/diagnostics/cpt_train_metrics.jsonl" \
  --metrics-jsonl "$RUN_DIR/diagnostics/cpt_eval_metrics.jsonl" \
  --output-dir "$RUN_DIR/eval/$(basename "$CKPT")" \
  --device cuda:0 \
  --attn-implementation sdpa \
  --vision-attn-implementation sdpa \
  --teacher-forced
```

独立 queue consumer 已接入 Merlin。H20 smoke 完成 checkpoint-10→20 后先提交 smoke
评测；它会消费两个 pending checkpoint，每任务至少 10 条 held-out，并要求 Base 与
checkpoint 的 teacher-forced CE 非空、inference error 为 0：

```bash
mlx job submitv2 --path locany_cpt_v4_h20x1_smoke_eval_merlin.yaml
```

formal 训练期间按 checkpoint 提交下面的单卡 job；Base 结果按 manifest/protocol 缓存，
同一验证集不会重复推理：

```bash
mlx job submitv2 --path locany_cpt_v4_h20x1_eval_merlin.yaml
```

也可在已经分配的单卡机器上消费指定 run：

```bash
RUN_DIR=/path/to/run DATA_DIR=/path/to/locany_cpt_v4_split_v2 \
  bash shell/run_locany_cpt_eval_merlin.sh h20
```

每个 checkpoint 输出 `predictions.jsonl`、`summary.json`、
`errors_by_task.jsonl`、`qualitative_samples.md`，以及 50 条 referring_kg 人工语义复核模板。
原始 prediction 与解析结果足以离线升级指标：

```bash
python scripts/recompute_locany_cpt_metrics.py \
  --predictions "$RUN_DIR/eval/checkpoint-STEP/predictions.jsonl" \
  --output-summary "$RUN_DIR/eval/checkpoint-STEP/summary.recomputed.json"
```

primary metrics 分别为 caption ROUGE-L、agent action type+point、agent grounding point
Hit@50、class-aware defect Macro-F1@0.5、all-elements one-to-one F1@0.5、single
grounding Recall@0.5、OCR one-to-one label-aware F1@0.5、referring/referring_kg
ROUGE-L、VQA 正确/错误 accuracy；Char-F1 均作为附属指标。Point 支持两种语法且坐标必须位于 `[0,1000]`；box
采用一对一最大总 IoU 匹配，UI defect 只允许同类匹配。综合分数仅为十任务 primary 的
等权 `heldout_task_macro_primary`。

best checkpoint 依次按 held-out task-macro primary 最大、task-macro main CE 最小排序；
关键弱任务相对历史 best 下降超过 3 个百分点时不自动标 best。少于完整十任务的 held-out
结果、train-pool 指标均不能成为 best。

分析 train–val 曲线：

```bash
python scripts/analyze_locany_cpt_curves.py \
  --train-metrics "$RUN_DIR/diagnostics/cpt_train_metrics.jsonl" \
  --eval-metrics "$RUN_DIR/diagnostics/cpt_eval_metrics.jsonl" \
  --output "$RUN_DIR/diagnostics/cpt_overfitting_analysis.json"
```

至少三个 milestone 才判定趋势；两点不足以声称“没有过拟合”。

## 7. JSON/JSONL 与 Excel

Source of truth 只有：

- `diagnostics/cpt_run_config.json`；
- `diagnostics/cpt_data_stats.json`；
- `diagnostics/cpt_train_metrics.jsonl`；
- `diagnostics/cpt_eval_metrics.jsonl`。

rank 0 原子写/追加这些文件。Excel 可随时离线重建：

```bash
python scripts/build_locany_cpt_excel.py \
  --diagnostics-dir "$RUN_DIR/diagnostics" \
  --output "$RUN_DIR/diagnostics/cpt_training_evaluation.xlsx"
```

工作簿严格只有 `Overview`、`TrainMetrics`、`EvalMetrics` 三个 sheet，冻结表头并启用筛选。
缺失值保持空白，不写成 0。`openpyxl` 缺失或保存失败只产生 warning，不会终止训练。

## 8. 验收

本地 CPT 单测：

```bash
python -m unittest \
  tests.test_cpt_split \
  tests.test_cpt_observability \
  tests.test_cpt_eval_metrics \
  tests.test_cpt_checkpoint_selection \
  tests.test_cpt_overfitting \
  tests.test_cpt_excel \
  tests.test_cpt_evaluator_end_to_end \
  tests.test_cpt_eval_queue \
  tests.test_cpt_smoke_validation \
  tests.test_locany_cpt \
  tests.test_locany_cpt_merlin
```

集群 smoke 还必须核对：20 step 正常、per-task sample/token/skip/CE 均存在、resume 后
计数单调不重复、A800/H20 schema 相同、多 rank reduce 不死锁、原始 TorchElastic 异常可见，
以及统计开启后 step time 增幅不超过 5%。这些结果必须来自真实 job 日志，不能由本地静态
检查代替。

两个 job 完成后可自动核对 checkpoint-10→20 resume、rank state、对账、单调性、十任务 CE、
eval queue、三表 Excel 和跨硬件 schema：

```bash
python scripts/validate_locany_cpt_smoke.py \
  --run a800=/path/to/a800-smoke-run \
  --run h20x2=/path/to/h20x2-smoke-run \
  --require-eval \
  --eval-samples-per-task 10 \
  --output /path/to/cpt_smoke_validation.json
```

`--require-eval` 还会确认最终 checkpoint 的 queue 状态为 completed、十任务
`eval_token_ce` 非空、推理错误为 0、task-macro 完整；启用 Excel 校验时还要求
`CPTEvalMetrics` 表至少包含十任务加一行 macro，而不只是存在空的 sheet。

常见失败含义：

- `DummyOptim ... param_groups`：旧版 CPT 自定义 optimizer 路径；当前兼容层不再直接假定
  DummyOptim 暴露 `param_groups`。
- `UI modules ... no parameter update`：旧版将 UI5 专属断言用于 CPT；当前仅在 UI5 模块
  真正启用时检查。
- `immutable run configuration changed`：resume 的采样/world/token/数据配置与 checkpoint
  不一致，需恢复原配置或换新输出目录。
- `missing dataloader_state_rank...`：checkpoint 不完整，禁止静默重启数据流。
- oversize warning：查 `pre_mtp_already_oversize` 与 `mtp_expansion_oversize`，不要直接截断答案。

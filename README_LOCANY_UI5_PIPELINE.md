# LocateAnything UI5 v4 统一训练—评测框架

本分支同时保证 UI5 v4 的 Detail Pyramid、Relation Query、Defectness Gate、PBD 与训练/推理基础设施使用同一条可诊断链路；不引入新的多任务算法，也不改变五类任务定义。

## v5 crop-only 数据与评测口径

当前正式数据模式为 `UI5_CROP_TRAIN_MODE=crop_only`，采样模式为
`UI5_UI_SAMPLING_MODE=task_balanced_all_records`。四个局部任务先使用测试侧同款 schema-v5
raw detector edge 生成 full-width 严格分区，再在训练侧仅删除穿过 GT 的 seam；因此 repair
后的结果仍无 overlap/gap。GT 只决定训练 crop 边界和局部标签，不会被绘制或传给模型。
`ui_content_missing` 继续使用原图 global view。所有纯负 strips 均保留，正样本原图中的无 GT
strips 也作为 negative record；partial strip 不会作为负样本。

`task_balanced_all_records` 为五个任务建立独立确定性 stream。较小 stream 只有在完整遍历后
才重复，较大 stream 不被下采样；自然正负分布不改写，manual repair fail-closed 保留。
训练启动及每 1000 step 写 `diagnostics/sampling_coverage_step_<N>.json`，其中合法但未进入
active pool 的记录数必须为 0。

评测拆成两个不可混用的集合：训练期间只使用 held-out validation cache，根据 raw Image/BBox
macro 选择 checkpoint，并仅在 validation 上冻结每任务 Gate 阈值；正式 1,555 张 test 只在
checkpoint 选定后运行一次。两者都使用 GT-free、strict non-overlap、raw-detector-edge-aligned
只读 cache，训练任务不会启动 PaddleOCR/icon worker。冻结 Gate 后会生成独立预测目录并重新
调用五任务 scorer，所以 gated BBox 指标不是 Image Gate 指标的复制。

从旧任务归档、生成 crop-only recipe、建立 validation cache、20+5 step resume smoke、正式
5000-step 训练和一次性 test 的完整无省略命令见 `README_UI5_COMMANDS.md` 的
“v5 crop-only F1 修复”章节。

Validation 自动早停默认关闭：提交参数 `--no-validation-early-stop` 对应
`EVAL_VALIDATION_EARLY_STOP=0`，周期评测和 Excel 记录仍会继续执行。只有显式传入
`--validation-early-stop` 时，pipeline 才会在 validation 连续两次 Image/BBox macro 均无改善后
停止；`EVAL_FAIL_POLICY` 仅控制评测失败，不控制该开关。

统一入口为：

```bash
python scripts/submit_locany_ui5.py --machine <a800|h20> --gpus <4|8> [options]
```

提交脚本从 `configs/locany_ui5_machines.json` 读取机器路径和 Merlin/Arnold 资源配置，渲染 `jobs/locany_ui5_merlin.template.yaml`，再执行 `mlx job submitv2 --path ...`。因此用户侧只有一个入口，同时不依赖 Merlin resource 字段是否支持运行时变量替换。

原文件 `locany_ui5_v3_a800x8_merlin.yaml` 的文件名属于历史命名；本工程和输出统一按 LocateAnything UI5 v4 定义。

## 常用命令

### A800 × 8 正式训练

```bash
python scripts/submit_locany_ui5.py \
  --machine a800 \
  --gpus 8 \
  --max-num-tokens 25600 \
  --enable-eval
```

### A800 × 4 正式训练

```bash
python scripts/submit_locany_ui5.py \
  --machine a800 \
  --gpus 4 \
  --max-num-tokens 12800 \
  --enable-eval
```

### H20 × 4 正式训练

```bash
python scripts/submit_locany_ui5.py \
  --machine h20 \
  --gpus 4 \
  --enable-eval
```

H20 × 4 使用 `magi + MAX_NUM_TOKENS=12800`；八卡仍为 25600。

### 关闭 evaluation 的四卡 smoke test

```bash
python scripts/submit_locany_ui5.py \
  --machine a800 \
  --gpus 4 \
  --max-num-tokens 12800 \
  --max-steps 2 \
  --disable-eval
```

该模式不会执行 step 0 或周期评测，直接调用原训练入口。

本轮 Relation/Gate/PBD 修复建议先跑 20-step 和 250-step 两级 smoke：

```bash
# 4 卡 20-step forward/backward 与保存加载基础检查
python scripts/submit_locany_ui5.py \
  --machine a800 --gpus 4 --max-num-tokens 12800 \
  --max-steps 20 --save-steps 20 --disable-eval \
  --run-name locany-ui5-v4-relationfix-a800x4-smoke20

# 4 卡 250-step Excel 检查；train_100steps 应只出现 100、200
python scripts/submit_locany_ui5.py \
  --machine a800 --gpus 4 --max-num-tokens 12800 \
  --max-steps 250 --save-steps 250 --disable-eval \
  --run-name locany-ui5-v4-relationfix-a800x4-smoke250

# 8 卡 20-step 使用同一 pipeline，仅 GPU 数和 token budget 不同
python scripts/submit_locany_ui5.py \
  --machine a800 --gpus 8 --max-num-tokens 25600 \
  --max-steps 20 --save-steps 20 --disable-eval \
  --run-name locany-ui5-v4-relationfix-a800x8-smoke20
```

只渲染和检查 YAML，不提交：

```bash
python scripts/submit_locany_ui5.py \
  --machine a800 \
  --gpus 4 \
  --disable-eval \
  --render-only
```

## 单 checkpoint 自动评测

直接提交一个只评测、不训练的四卡任务：

```bash
python scripts/submit_locany_ui5.py \
  --machine a800 \
  --gpus 4 \
  --eval-checkpoint /path/to/checkpoint-4000 \
  --eval-step 4000 \
  --relation-gate-mode observe \
  --eval-max-images-per-task 10
```

`--eval-max-images-per-task` 只用于小样本 smoke；省略或设置为 0 时仍跑完整的每任务 1,555 条评测。

复评旧 relationfix checkpoint-1000（不会用 0.5 提前清空生成）：

```bash
python scripts/submit_locany_ui5.py \
  --machine a800 --gpu 4 \
  --run-name locany-ui5-v4-relationfix-a800x4 \
  --eval-checkpoint /path/to/relationfix/checkpoint-1000 \
  --eval-step 1000 \
  --relation-gate-mode observe
```

如果已经处于分配好四张 GPU 的节点，也可以直接执行 pipeline：

```bash
MACHINE_TYPE=a800 \
GPU_COUNT=4 \
CUDA_DEVICES=0,1,2,3 \
EVAL_GPU_DEVICES=0,1,2,3 \
PIPELINE_MODE=eval \
EVAL_CHECKPOINT=/path/to/checkpoint-4000 \
EVAL_STEP=4000 \
bash shell/run_locany_ui5_pipeline.sh
```

它会依次完成：

```text
checkpoint 校验和 patch
→ 五任务四卡并行推理
→ qwen3vl merge + score
→ evaluation_history.json/csv
```

若评分工程不在训练仓库中，增加：

```bash
SCORER_ROOT=/path/to/qwen3vl
```

`SCORER_ROOT` 中必须存在 `qwen3vl_merge_and_score_fixed_5tasks.py`。

## 最终参数与优先级

提交参数会写入 Merlin `envsList`，pipeline 启动后再次解析并打印最终值。显式参数优先于机器默认值。

关键参数：

```text
MACHINE_TYPE
GPU_COUNT
CUDA_DEVICES
DATA_VERSION
VERSION
MAX_STEPS
WARMUP_STEPS
LEARNING_RATE
MAX_SEQ_LENGTH
MAX_NUM_TOKENS_PER_SAMPLE
MAX_NUM_TOKENS
SAVE_STEPS
ENABLE_EVAL
EVAL_AT_START
EVAL_INTERVAL_STEPS
EVAL_GPU_DEVICES
EVAL_FAIL_POLICY
SCORER_ROOT
```

默认值：

| 配置 | Attention | 单样本上限 | packed batch 上限 |
|---|---:|---:|---:|
| A800 × 8 | sdpa | 7268 | 25600 |
| A800 × 4 | sdpa | 7268 | 12800 |
| H20 × 4 | magi | 8192 | 12800 |

`MAX_NUM_TOKENS` 在当前 `StreamPackedDatasetMTP` 中是每个 rank 的 packed batch token 硬上限。它不是四卡或八卡的全局 token 总和。`MAX_NUM_TOKENS_PER_SAMPLE` 先过滤单条原始样本，随后 packer 用 `MAX_NUM_TOKENS` 决定一个 rank 上何时 flush batch。

因此四卡默认 12800 是保守起点，不代表 25600 必然 OOM。需要分别运行：

```bash
# A800 × 4 / per-rank packed budget 25600
python scripts/submit_locany_ui5.py \
  --machine a800 --gpus 4 --max-num-tokens 25600 \
  --max-steps 2 --disable-eval

# A800 × 4 / per-rank packed budget 12800
python scripts/submit_locany_ui5.py \
  --machine a800 --gpus 4 --max-num-tokens 12800 \
  --max-steps 2 --disable-eval
```

训练脚本会生成 `gpu-memory-*.csv`，并在退出时汇总单卡峰值显存。正式将哪个值作为默认值，应以这两次 A800 实测为准。

## 周期评测和 checkpoint 生命周期

默认正式配置：

```text
MAX_STEPS=16000
SAVE_STEPS=4000
ENABLE_EVAL=1
EVAL_AT_START=1
EVAL_INTERVAL_STEPS=1000
EVAL_FAIL_POLICY=stop
```

实现不会把 Trainer 的正式 `SAVE_STEPS` 改成 1000，也不会为每次 segment 改变 `MAX_STEPS`。训练始终使用相同的总 `MAX_STEPS=16000` 创建 cosine scheduler，只通过 `SegmentStopCallback` 在绝对 global step 1000、2000、3000……设置：

```text
should_save=True
should_training_stop=True
```

每次 resume 均恢复：

```text
model
optimizer
scheduler
global step
random state
dataloader state
```

这样不会因为每段使用不同 `max_steps` 而改变学习率轨迹。

checkpoint 策略：

```text
checkpoint-4000、8000、12000、16000：永久保留
其他 1000-step checkpoint：临时评测/resume checkpoint
```

只有当下一 checkpoint 通过 resume 完整性校验且评测成功后，才会清理更旧的临时 checkpoint。任意时刻至少保留最近一个可恢复 checkpoint 和所有正式 checkpoint。`EVAL_FAIL_POLICY=warn` 下评测失败时不清理临时 checkpoint。

短周期状态机验证：

```bash
python scripts/submit_locany_ui5.py \
  --machine a800 \
  --gpus 4 \
  --max-num-tokens 12800 \
  --max-steps 50 \
  --save-steps 40 \
  --eval-interval-steps 10 \
  --enable-eval
```

预期评测 step 为 0、10、20、30、40、50。

## 五任务并行推理

默认 `EVAL_INFERENCE_CROP_MODE=detector_scan` 且
`EVAL_DETECTOR_CACHE_MODE=readonly`。PP-OCRv5 与 OmniParser 必须在训练前离线完成；训练中的
step-0 和周期评测只校验并读取缓存，不再进入 detector pipeline。缓存目录为：

```text
${EVAL_DETECTOR_CACHE}/
  manifest/
  detections/{text,icon,merged}/
  horizontal_scan_v5_raw_detector_edge_aligned/detector_scan_crops.jsonl
  horizontal_scan_v5_raw_detector_edge_aligned/{summary.json,statistics.csv,eval_detector_cache_ready.json}
  horizontal_scan_v5_raw_detector_edge_aligned/gallery/index.html
```

离线构建时主进程和 icon worker 使用 LocateAnything Python，text worker 显式使用独立的
UI5PaddleOCR Python；不要在两个环境之间互装 Torch/Paddle。两个 runtime preflight 均在 GPU
worker 前执行。schema-v5 几何只读取原始 text/icon bbox 和图片尺寸，并采用半开区间 `[y1,y2)`：
相邻 core 首尾严格相接，既无遗漏也无重复像素行。seam 候选直接取原始 bbox 的 `y1/y2`；
凡落入任意其他 bbox 严格内部的 raw edge 都会被全局剔除。生产模式 guard 固定为 0，禁止从
guarded/protected band、空隙中点或理想等分点生成 seam。安全 raw edge 不足时减少 crop 数量，
最差保留完整原图，不允许 balanced fallback、穿框、context 扩张或 overlap。每个 crop 横向
贯穿完整宽度，每个原始 detector bbox 都唯一完整归属一张 crop。
评测/推理禁止读取训练 GT repair；`content_missing` 单独保留完整原图视图。

`eval_detector_cache_ready.json` 在输入、detector shard、merged、几何 manifest、summary、CSV 和
gallery 全部写入并通过门禁后最后原子生成。readonly 会重新校验 dataset/detector/geometry
digest；缓存缺失、不完整或 digest 不一致时 fail closed，不回退现场 detector 或 full image。
正式评测还要求 `cache_scope=full_test`、`max_images_per_task=0`、1,555 张内容唯一图片，并拒绝
schema-v4 或 preview marker。更改几何时使用新的 `EVAL_SCAN_NAME`，只重跑 CPU crop；raw
detector shard 保持不变。

正式运行前建议先按 [README_UI5_COMMANDS.md](README_UI5_COMMANDS.md) 的单命令预览方式选择
至少 200 张测试图片，按 sparse/medium/dense 检查 gallery 和统计。完整离线 cache/validator
命令见 [README_UI5_COMMANDS.md](README_UI5_COMMANDS.md)。不得在正在训练使用的共享环境中
现场安装或替换 Torch/Paddle 依赖。

任务：

```text
occlusion
cropping
text_overflow
text_ellipsis
content_missing
```

`run_ui5_parallel_inference.py` 启动四个固定物理 GPU worker。每个子进程通过：

```text
CUDA_VISIBLE_DEVICES=<physical GPU>
--device cuda:0
```

隔离设备。主进程不硬编码某两个任务共享 GPU：

1. 首次评测统计五个 JSONL 的记录数，以此估算耗时并采用 longest-processing-time-first 排队。
2. 实际运行后把各任务耗时写入 `evaluation/task_runtime_profile.json`。
3. 后续 checkpoint 根据真实耗时重新排序；任意 GPU 完成后领取队列中的下一任务。

每个 worker 有独立日志和 summary。任一 worker 非零退出、没有生成预测 JSON，或有任务未启动，主进程都会返回非零并打印 step、task、GPU、命令、exit code 和日志路径。

## 自动评分与输出

每个 step 的预测目录（默认 detector-scan）：

```text
${OUTPUT_DIR}/inference-checkpoint-${STEP}-detector-scan/
```

原始评分结果：

```text
${OUTPUT_DIR}/evaluation/raw/checkpoint-${STEP}-${TIMESTAMP}/
```

总历史：

```text
${OUTPUT_DIR}/evaluation/evaluation_history.json
${OUTPUT_DIR}/evaluation/evaluation_history.csv
```

history 包含：step、机器类型、训练 GPU 数、`MAX_NUM_TOKENS`、checkpoint、五类 image/bbox 指标、image/bbox macro precision/recall/F1、起止时间和状态。`macro_precision/recall/F1` 列默认对应主要的 Image Macro 指标，同时保留明确的 `image_macro_*` 与 `bbox_macro_*` 列。

## 两-sheet 训练/评测诊断

唯一诊断工作簿为：

```text
${OUTPUT_DIR}/diagnostics/ui5_training_evaluation.xlsx
```

它严格只包含：

```text
train_100steps：每 100 optimizer global step 的窗口 loss、五任务 Gate 指标、三层 Detail/Relation/PBD 输出与梯度
eval_1000steps：checkpoint-0 及每次周期评测的五任务 image/bbox 与两行 macro
```

rank 0 使用临时 `.xlsx` 保存并重新打开校验，成功后才以 `os.replace` 覆盖正式文件；各 rank 的隐藏窗口状态文件用于在 100-step 窗口中途 resume，已有 step 不会重复追加。每图 Gate 诊断写入预测任务目录下的 `gate/`，汇总写入 `_gate_metrics.json`，包括 `p_defect`、Gate precision/recall/F1、过滤数量及 Gate 后 FP。

运行环境需要 `openpyxl>=3.1`（已加入 `pyproject.toml`）；pipeline 会在加载模型前检查并给出明确错误。

评测默认使用 `relation_gate_mode=observe`：始终执行原始 bbox 生成，同时保存 image-level `p_defect`，再用同一批 raw prediction 离线扫描 0.00–0.60。完整扫描保存在当前预测目录和评分目录的 `gate_threshold_sweep.json/.txt`；`t=0` 严格等价于 raw/no-hard-gate。只有显式指定 `--relation-gate-mode hard` 时，低于阈值才提前返回 `<box>none</box>`。

训练开始前会导出真正包含 Relation、image Gate、slot Gate 和 PBD 权重的 full-model `checkpoint-0`，因此 step 0 与后续 checkpoint 结构一致。旧 relationfix checkpoint 可用 observe 兼容模式复评；其历史 slot gate 只用于复现，不作为新 image Gate 的训练结论。

checkpoint 之间的四组参数更新可独立复核并输出 JSON/文本：

```bash
python scripts/check_ui_relation_training_state.py \
  --checkpoint-0 "${OUTPUT_DIR}/checkpoint-0" \
  --checkpoint-n "${OUTPUT_DIR}/checkpoint-1000" \
  --output-json "${OUTPUT_DIR}/diagnostics/ui_relation_update_step1000.json" \
  --output-txt "${OUTPUT_DIR}/diagnostics/ui_relation_update_step1000.txt"
```

训练模式在 resume 前执行严格权重组审计；缺少新的 image Gate 的旧 checkpoint 只能使用上面的 evaluation-only observe 命令复评，不能静默续训。

对照无 Relation 路径可运行：

```bash
python scripts/inference_ui_defect_locany.py ... --enable-ui-relation false
```

4 卡和 8 卡使用同一 pipeline，并保留原训练设置：四卡 `GRADIENT_ACCUMULATION_STEPS=2`，八卡为 `1`；其余训练参数由提交器做一致性检查。默认输出目录包含 `a800x4/a800x8`，避免两种运行互相覆盖。

## Checkpoint patch

单独调用：

```bash
python scripts/patch_locany_checkpoint.py \
  --base-model "${BASE_MODEL}" \
  --checkpoint "${CHECKPOINT}" \
  --project-root "${PROJECT_ROOT}"
```

默认行为是缺失则复制、存在则跳过；重复执行是幂等的。若需要用当前仓库的 LocateAnything 推理代码覆盖旧文件：

```bash
python scripts/patch_locany_checkpoint.py \
  --base-model "${BASE_MODEL}" \
  --checkpoint "${CHECKPOINT}" \
  --project-root "${PROJECT_ROOT}" \
  --force
```

自动评测使用 `--force`，保证 checkpoint 中的 `modeling_locateanything.py` 和 `relation_modules.py` 与本次代码快照一致。
同时使用 `--validate-relation-weights` 检查 Relation Pyramid、Gate Heads 和 PBD 的权重组确实存在；缺失时在加载推理模型前直接失败。

## Fail-fast 检查

pipeline 会依次检查：

```text
机器配置和 GPU 列表
Python/基础模型/训练数据/评分脚本
训练进程返回码
checkpoint 模型权重、trainer state、DeepSpeed/optimizer state、rank state
patch 返回码
五个 inference worker 返回码和预测文件
qwen3vl 返回码和 all_tasks_evaluation.json
history 写入结果
```

正式训练默认 `EVAL_FAIL_POLICY=stop`。`warn` 仅用于明确接受评测失败但继续训练的调试场景。

# LocateAnything UI5 v4 统一训练—评测框架

本框架只改训练、checkpoint 和评测基础设施，不修改 LocateAnything UI5 v4 的模型结构、Relation 模块或五类任务定义。

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

H20 × 4 沿用现有配置的 `magi + MAX_NUM_TOKENS=25600` 默认值，可通过 `--max-num-tokens` 覆盖。

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
  --eval-step 4000
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
| H20 × 4 | magi | 8192 | 25600 |

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

每个 step 的预测目录：

```text
${OUTPUT_DIR}/inference-checkpoint-${STEP}-full/
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

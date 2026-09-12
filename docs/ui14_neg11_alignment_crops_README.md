# neg11：对齐任务 crops 训练和推理

本实验使用已经完成的 neg11 `available` 数据：不重新收集正常图、不重新划分，
七项合成任务继续在两侧标签存在时按 1:1 采样，test 使用实际可用数量。
`ui_alignment` 改为 detector crops 训练和推理；`content_missing` 保持全图。
其余 12 项的输入策略不变。保持 14 个任务和同一 M32，原始 GT 像素框不再投影一次。
裁剪几何只依赖 detector；GT 用于裁片标签和覆盖报告，评分仍将预测合并回原图。

新 run 从 CPT checkpoint-9000 初始化新的 SFT optimizer/scheduler，训练 16k steps。
4×A800、每 100 步训练记录、step 0 和每 1000 步 14 项评测/36 行、4k 正式保存、
UI5 best 口径和 EVAL_FAIL_POLICY=stop 保持原值。
生成使用旧正式配置 `legacy + hybrid`，SDPA、视觉 flash_attention_2、PBD、observe gate。
全部 14 项评测任务每张物理 GPU 只有一个推理进程，四卡最多同时运行四个任务，完成后继续领取。
正式配置为 `EVAL_INFERENCE_WORKERS_PER_GPU=1`；旧 YAML/环境/命令中的值 2 兼容接收并按 1 执行。
直接调用并行推理脚本和旧 checkpoint 独立补评也遵循同一限制。
独占卡消除本调度器内的同卡 worker 竞争，不是对任意超大单样本不会 OOM 的保证。

## 评测中断后的更新与续跑

等本次失败作业及它的评测 worker 退出后，在原 alignment-crops checkout 更新并重新提交。
其他仍在运行的对照实验继续使用自己的目录。此次只调整评测调度和日志，无需重做数据准备：

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui14-neg11-alignment-crops
git pull --ff-only origin codex/ui14-neg11-alignment-crops-v1 &&
bash shell/ui14_neg11_alignment_crops_a800.sh submit --resource-group aiai_locate
```

继续使用相同 OUTPUT_DIR、checkpoint 和预测目录，不传 --overwrite。入口检查最新完整
checkpoint 的 optimizer/scheduler/rank 状态，再补齐该步评测后继续训练。
本次 checkpoint-3000 不因评测 OOM 被清理；通过完整恢复校验后，无需重训前 3000 步。
任务内已完成图片仍按原推理身份核对后跳过，OOM/缺失图片重试；全部图片完成的任务不加载模型。
所有任务均独占卡，旧 `EVAL_EXCLUSIVE_GPU_TASKS` 子集不会使其他任务恢复共卡。
这只是推理并发变更，不改变图片、权重、prompt、输入策略和解码身份。不要传 `--overwrite`，
不要删除 checkpoint、预测目录或 Excel，无需重跑 prepare/cache/finalize。

每个 task 的推理子进程成功退出后立即评分，由唯一 CPU 协调进程原子写入
`diagnostics/ui5_training_evaluation.xlsx` 的 `eval_1000steps`：
每任务 image/bbox 两行；`evaluation_status=partial` 表示整轮未齐，失败时保留这些行并标为 failed。
UI5 的原 no_figma/class_id 评分器、UI9 的原 IoU=0.1/非法输出处理均不变，bbox 没有 TN。
其他任务失败后仍在执行的 worker 完成时，其结果也会保存。

每 10 秒显示成功/失败数、仍在运行的 task/GPU、尚未启动任务、失败原因和日志路径。
若已发生终止性失败，明确打印队列停止派发、正在等待其他 worker 完成；任务失败事件不算成功数。
末尾异常再次带上失败 task/原因，不再只显示 subprocess exit 1。
`parallel_inference_progress.json` 位于本步预测目录，包含当前 running/pending/stopped/tasks；
评分 attempt 的 `completions/_scheduler/status.json` 保存同一状态。
`EVAL_FAIL_POLICY=stop` 继续生效，未齐 14 项不得继续训练，不自动放宽评分或忽略失败。

逐任务明细在输出目录的
`evaluation/partial/ui14-step-<step>-<attempt>/<task>/metrics.json`，包含本轮身份、真实计数及 gate 指标；
该 attempt 下的 `completions/` 保存调度器完成记录。
`evaluation/ui14-step-<step>.json` 持续更新已评分任务和失败原因。
只有完整 14 项成功后才写入全部 36 行及 UI5/UI9 macro/micro、best/history 并允许继续训练。
部分任务不能代表完整评测；续跑按任务行更新，不累加重复行。各次评分报告分别保留。

| 内容 | 路径（前缀均为 `/mnt/bn/intelligent-service-yg/logging/sicheng_workspace`） |
| --- | --- |
| 新 checkout | `code/Eagle_LocateUI5_v4/Embodied-ui14-neg11-alignment-crops` |
| 只读父数据 | `gui_data/ui14_cpt9000_neg11_v1` |
| 新数据/清单/对齐缓存 | `gui_data/ui14_neg11_alignment_crops_v1` |
| 新训练输出 | `gui_models/locany-m32-cpt9000-ui14-neg11-alignment-crops-a800x4-v1` |

新入口不使用旧 UI14_DATA_ROOT / UI14_ALIGNMENT_DATA_ROOT 的默认值。
自定义新数据位置使用 `UI14_ALIGNMENT_CROPS_DATA_ROOT`，父数据位置使用
`UI14_ALIGNMENT_CROPS_PARENT_DATA_ROOT`。旧训练、源码、checkpoint 和 Excel 不修改。
数据的 normalization_id、selection 和原图 test 的 eval_set_id 保持冻结；
新的任务 view_policy 随模型配置和评测身份保存，不与旧预测混用。

## 新 checkout 与资源分阶段命令

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4
git clone --single-branch --branch codex/ui14-neg11-alignment-crops-v1 \
  https://github.com/YoungSeng/Embodied.git Embodied-ui14-neg11-alignment-crops
cd Embodied-ui14-neg11-alignment-crops

# CPU：冻结既有 neg11 数据，只为对齐任务准备新增缓存。
bash shell/ui14_neg11_alignment_crops_a800.sh prepare &&
UI14_PREPARE_WORKERS=16 bash shell/ui14_neg11_alignment_crops_a800.sh cache-prepare
```

prepare 的成功标志为 `stage_summaries/prepare.json` complete；此时 CPU 报告 ready=false，
等待对齐缓存。cache-prepare 完成 `cache/ui_alignment/{train,test}/manifest/ui14_prepare_ready.json`。
父版本规范化文件原样保留；本次缓存的本地输入指针另存 `task_input_manifest.json`。

```bash
# 独立四卡 GPU 资源：仅新增 ui_alignment train/test 的 OCR + OmniParser。
# 正在训练的 neg11 任务继续运行；此处使用另一份资源。
bash shell/ui14_neg11_alignment_crops_a800.sh cache
```

cache 复用自身已完成 detector 分片，不做其他任务的 CPU 扫描或裁图。
成功后立即释放这份预处理 GPU，返回 CPU 节点：

```bash
UI14_CROP_WORKERS=16 UI14_PNG_COMPRESS_LEVEL=1 \
  bash shell/ui14_neg11_alignment_crops_a800.sh cache-finalize &&
bash shell/ui14_neg11_alignment_crops_a800.sh finalize &&
bash shell/ui14_neg11_alignment_crops_a800.sh check &&
bash shell/ui14_neg11_alignment_crops_a800.sh submit --resource-group aiai_locate
```

cache-finalize 使用原有水平扫描规则、单原图完成索引、原子低压缩 PNG 和有界线程池；
支持续跑。finalize 只消费完成缓存，输出 `alignment_crop_check.json`、`training_recipe.json`、
`evaluation_manifest.json`、`sampling_stats.json`、ready=true 的 `cpu_check_report.json` 和
`formal_job.yaml` / `formal_runtime.json`。其他任务的 PNG/标签/detector 只读引用父版本。
统计取实际产物，包括原图数、正负 crop 数、GT 覆盖和采样数；不使用历史数量代替。

submit 使用 `--machine a800 --gpus 4` 和现有 mlx submitv2，
`ies_aiai_experience/AIAI_locate`、group_id=2146，
queue=`compute-3302-yg-cloudnative-ai-aiai.locate-guarantee`。
新 Excel 为新训练输出下的 `diagnostics/ui5_training_evaluation.xlsx`。
参考正式 YAML 随代码保存在 `docs/ui14_neg11_alignment_crops_a800.yaml`；
实际提交必须通过本批数据的 finalize/check。

```bash
bash shell/ui14_neg11_alignment_crops_a800.sh status
```

## 已准备旧 ui_alignment 权重的独立补评切回正式解码

从本次新 checkout 执行以下旧补评入口，它继续使用此前 alignment_context 数据目录和快照，
不会修改正在运行的 neg11 训练。默认旧 step-2000/4000 和各自对应旧 test 的原全图视图；
这与上面的新 crops 训练实验分开记录。

```bash
bash shell/ui14_alignment_context_a800.sh eval-prepare \
  --task ui_alignment --steps 2000 4000 --answer-grammar legacy

# 上一步成功后，在独立评测 GPU 资源上运行。
bash shell/ui14_alignment_context_a800.sh eval-existing \
  --task ui_alignment --steps 2000 4000 --answer-grammar legacy --gpu-devices 0
```

匹配的模型快照文件直接复用。切换 grammar 会产生新的 comparison_id/预测目录，
不复用结构约束版本的预测。原 raw 审计可继续复用。
如果曾显式指定 --old-run / --old-manifest / --data-root，继续传入相同路径。

当前环境不能连接 A800；本地 CPU 回归不表示真实 detector、补评或提交已执行。

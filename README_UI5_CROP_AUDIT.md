# UI5 Crop v4.1：GT 修复、训练接线与无遗漏推理切图

当前分支是 `locany-ui5-det-crop-v1`，CPT 数据与 CPT 训练入口保持关闭。v4.1 只读取已经
完成的 17,281 张唯一图片 manifest、OCR/icon JSONL 和 `crop_audit_v3`，不会重新运行
PP-OCRv5、OmniParser，也不会覆盖 `crop_audit_v3` 或原始 `detections.jsonl`。

当前代码提交：

- v4.1 主实现：`a0ee63abbf7ec702a5891e4f9bb4bcd1b5a02dbb`；
- 实时进度与 ETA：`b353b6d2c41a1751dab3bee8bb1a5cb7bd9ea727`；
- 从 `locany-cpt-v1` 功能等价移植的运行时修复来源：
  `de30357f73b6b393840efcb7eb3ca37182e86cc4`；
- 从 `locany-cpt-v1` 功能等价移植的 SIGBUS/checkpoint 修复来源：
  `ed5add660608b93ed4f9d5c68efb1c04478aa6bd`。

这里没有整体 merge 或覆盖 `locany-cpt-v1`。crop audit、training-ready、recipe、运行时
preflight 和 checkpoint 完整性逻辑均已合并保留。`ui-region-parser` 仍是同级独立仓库，固定在
`06eaebf8eb4ea01e61b690f2ff972bf614915918`。

## 现在只需要先跑这一条

在 A800/YG 内网机器的 crop worktree 中执行：

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop

UI5_OUTPUT_ROOT=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_crop_audit_20260825
UI5_BASE_META=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied/data/ui_defect_locany_v3/recipe/ui_defect_5class_train.json

bash shell/run_ui5_gt_repair.sh \
  --output-dir "${UI5_OUTPUT_ROOT}" \
  --parser-root ../ui-region-parser \
  --base-meta "${UI5_BASE_META}" \
  --source-audit-name crop_audit_v3 \
  --crop-audit-name crop_audit_v4_gt_repair \
  --expected-unique-images 17281 \
  --resume
```

旧工程中的 base meta 只作为只读输入，不需要复制整个 `data/`，也不要 `mkdir data` 后期待
脚本自动补齐数据。若新 worktree 已经有同一份 `data/ui_defect_locany_v3`，可以把
`UI5_BASE_META` 改为新 worktree 内的绝对路径。

这一条 wrapper 会按顺序完成：

1. 校验 v3、17,281 个 image id、原始 detection digest 和 107 个原始失败；
2. 排除唯一已确认错误标注，并保存独立证据目录；
3. 对 106 个有效失败执行仅限训练集、仅限对应 task/sample 的 GT repair；
4. 重跑 CPU geometry，复用或生成普通矩形 crop，生成 106 张四联图；
5. 写 summary、CSV、Excel 和 task-aware manifest；
6. 生成 full-only 与 full+crop recipe；
7. 所有 19 项 gate 通过后，最后原子写入 `training_ready.json`。

它不会启动训练。它也不会调用 prepare/text/icon/merge；开头应打印：

```text
detector_stages_executed=[]; OCR/icon/merge disabled
```

## 实时看进度和剩余时间

当前终端会每 10 秒打印完成数、百分比、速度、耗时和 ETA，依次显示：

```text
gt-repair-geometry
gt-repair-materialize
gt-repair-visualizations
gt-repair-reports
gt-repair-recipe
```

`--resume` 开始时还会打印已复用的 geometry shard、crop 和四联图数量。另一个终端可查看
最后一次原子状态快照：

```bash
watch -n 2 'python -m json.tool \
  work_dirs/ui5_crop_audit_20260825/crop_audit_v4_gt_repair/run_status.json'
```

ETA 只按本轮真正需要处理的未完成项计算。刚启动或全部复用时速度为 0，ETA 显示
`--:--:--` 或很快完成是正常现象。

## v4.1 输出与指标语义

新结果只写入：

```text
work_dirs/ui5_crop_audit_20260825/crop_audit_v4_gt_repair/
  excluded_annotation_cases/
  gt_repair_visualizations/
  candidate_TA_CTX015_H050_GT_REPAIR/
  training_recipes/
  gt_repair_detections.jsonl
  gt_repair_actions.jsonl
  excluded_training_samples.jsonl
  task_aware_manifest.jsonl
  materialization_summary.json
  summary.json
  statistics.csv
  ui5_crop_audit.xlsx
  training_ready.json
```

报告必须同时保留两套口径：

- `raw_detector_region_gt_recall = 17691 / 17798 = 99.3988%`，这是未使用 GT repair 的
  原始 detector crop 结果；
- `training_materialization_gt_recall_after_repair = 17797 / 17797 = 100%`，只表示排除
  1 个错误标注后，训练物化 crop 完整覆盖清洗后的训练 GT；它不代表 OCR/icon detector
  具有 100% 泛化召回率。

`sample_3a3922c5762298f04c8d` 只从 `ui_text_overflow` 训练记录中排除；同一图片在其他任务的
监督仍保留。GT repair 严禁进入 val、test 和真实推理。

## 如需单独重建 recipe

上面的 wrapper 已经自动执行本节。只有 audit 已完成、仅需重建 recipe 时才单独运行：

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop

UI5_PROJECT_ROOT=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop
UI5_AUDIT_DIR=${UI5_PROJECT_ROOT}/work_dirs/ui5_crop_audit_20260825/crop_audit_v4_gt_repair

python scripts/build_ui5_crop_training_recipe.py \
  --audit-dir "${UI5_AUDIT_DIR}" \
  --base-meta "${UI5_PROJECT_ROOT}/data/ui_defect_locany_v3/recipe/ui_defect_5class_train.json" \
  --task-aware-manifest "${UI5_AUDIT_DIR}/task_aware_manifest.jsonl" \
  --excluded-samples "${UI5_AUDIT_DIR}/excluded_training_samples.jsonl" \
  --mode full_plus_crop \
  --output-dir "${UI5_AUDIT_DIR}/training_recipes" \
  --require-valid-gt-recall 1.0

test -s "${UI5_AUDIT_DIR}/training_recipes/ui_defect_5class_train_full_plus_crop.json"
test -s "${UI5_AUDIT_DIR}/training_recipes/ui_defect_5class_train_full_plus_crop.jsonl"
test -s "${UI5_AUDIT_DIR}/training_recipes/recipe_summary.json"
test -s "${UI5_AUDIT_DIR}/training_ready.json"
```

如果新 worktree 没有 `data/ui_defect_locany_v3`，仍应把 `--base-meta` 换成旧工程中的绝对
路径，不要复制内网数据。

## 运行状态说明

代码和单元测试已在本地完成；公司内网的 v4 全量 crop、17,797 GT 验收、20-step 四卡
训练 smoke 和正式训练仍必须由上述集群命令实际执行。本 README 不把本地代码测试冒充为
A800 实跑结果。完整 smoke 和正式训练命令见
[README_UI5_COMMANDS.md](README_UI5_COMMANDS.md)。

---

## 以下是 v3 原始检测审计历史说明

以下章节保留用于追溯最初 OCR/icon 和 A/B/C、task-aware 参数审计。已有 detection 完整时，
执行 v4.1 不需要重新运行这些命令。

## 1. 一次性准备仓库

```bash
cd /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Eagle/Embodied
git fetch origin
git worktree add ../Embodied-ui5-det-crop \
  -b locany-ui5-det-crop-v1 \
  origin/locany-cpt-v1

cd ..
git clone https://github.com/YoungSeng/ui-region-parser.git
git -C ui-region-parser checkout 06eaebf8eb4ea01e61b690f2ff972bf614915918
```

如果 worktree 和 parser 已经存在，不要重复执行这一节。

## 2. 数据目录：直接传绝对路径，不要复制

`--locany-data-dir data/ui_defect_locany` 是相对于当前 worktree 的示例。如果新 worktree
没有这个目录，执行 `mkdir data` 只会得到空目录，不能解决问题，也不要把整份旧工程复制到
新 worktree。

现有训练 JSONL 可以继续放在旧工程的数据目录中；审计脚本只读它们。A800/YG 集群可直接用：

```bash
SOURCE_DIR=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data
LOCANY_DATA_DIR=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied/data/ui_defect_locany_v3

test -d "${SOURCE_DIR}"
test -d "${LOCANY_DATA_DIR}"

for TASK in ui_occlusion ui_cropping ui_text_overflow ui_text_ellipsis ui_content_missing; do
  test -f "${LOCANY_DATA_DIR}/${TASK}_train.jsonl" || {
    echo "缺少 ${LOCANY_DATA_DIR}/${TASK}_train.jsonl" >&2
    exit 1
  }
done
```

该目录里的 `recipe/`、`conversion_summary.json` 可以保留原地；本阶段主要读取五个
`*_train.jsonl`，存在 `*_val.jsonl` 时还会用于检查相同内容是否跨 train/val。把
`ui_defect_locany_v3` 作为数据输入，不代表代码从旧 v4 分支开始；新 worktree 的 Git base
仍是当前 `locany-cpt-v1`。

不要把内网数据传到本机或提交 Git。只要审计命令也在公司内网集群执行，传绝对路径即可。

## 3. 模型权重

### PP-OCRv5：默认自动下载，不需要手动放权重

不传 `--text-model-dir` 时，脚本使用 `PP-OCRv5_server_det`，PaddleOCR 会在首次 text
阶段自动下载到自己的缓存目录，后续直接复用缓存。

如果希望在四卡任务启动前先单进程预热缓存，可选执行：

```bash
cd ../ui-region-parser
export PADDLE_PDX_CACHE_HOME=/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/cache/paddlex
python -c "from paddleocr import TextDetection; TextDetection(model_name='PP-OCRv5_server_det', device='cpu')"
```

这只是提前触发自动下载，不是额外必需权重。正式命令不用传
`--text-model-dir`。如果集群已经有本地 OCR 模型，才使用：

```bash
--text-model-dir /absolute/path/to/PP-OCRv5_server_det_infer
```

### Paddle PIR / OneDNN 报错

如果日志出现：

```text
ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttributeAttribute]
```

说明当前 Paddle/PIR 组合进入了 MKLDNN/OneDNN 路径，并非图片或 GPU 损坏。审计脚本默认
`enable_mkldnn=false`；不要添加 `--enable-mkldnn`。更新代码后直接重新执行原来的
`--stage all --resume` 命令即可：已经完成的 prepare 会经过完整性验证后跳过，不会重新扫描
17,281 张图片；尚未完成的 text shard 会用关闭 MKLDNN 的配置重新运行。

只有明确验证过兼容的 Paddle CPU 环境才考虑传 `--enable-mkldnn`，四卡 GPU 正式检测不需要。

### OmniParser icon detector：需要手动下载

`huggingface-cli` 已废弃。集群已经提示 `hf` 可用时，直接使用现成的 `hf`，不要在
LocateAnything、parser 或训练环境里执行 `pip install -U huggingface_hub`。升级
`huggingface_hub` 可能改变现有 `transformers` / `tokenizers` 的依赖组合。

只下载需要的 v3 detector 文件：

```bash
cd ../ui-region-parser
command -v hf
mkdir -p weights/icon_detect_v3
hf download microsoft/OmniParser-v2.0 \
  icon_detect_v3/model.pt \
  --revision refs/pr/37 \
  --local-dir weights
test -f weights/icon_detect_v3/model.pt
```

如果 `command -v hf` 找不到命令，不要修改现有 conda 环境；使用一次性的独立 venv：

```bash
python3 -m venv /tmp/ui5-hf-cli
/tmp/ui5-hf-cli/bin/python -m pip install huggingface_hub

cd ../ui-region-parser
mkdir -p weights/icon_detect_v3
/tmp/ui5-hf-cli/bin/hf download microsoft/OmniParser-v2.0 \
  icon_detect_v3/model.pt \
  --revision refs/pr/37 \
  --local-dir weights
test -f weights/icon_detect_v3/model.pt
```

临时 venv 只用于下载文件，不参与 Paddle、Torch 或 LocateAnything 运行，因此不会改动当前
环境中的 `huggingface_hub`、`transformers` 或 `tokenizers`。

最终必须得到：

```text
../ui-region-parser/weights/icon_detect_v3/model.pt
```

该命令来自 Microsoft OmniParser 官方说明，只下载 icon detector，不下载 caption 模型或
整个模型仓库。也可以通过 `--icon-model /absolute/path/model.pt` 指定已有文件。

### `icon_detect_v3 requires torch and torchvision` 报错

这条短错误不一定表示 `torch` 和 `torchvision` 都没装。固定版本 parser 会把下面几种情况
统一包装成同一句话：缺少 `torchvision`、Torch 与 torchvision 二进制不匹配、
`torchvision::nms` 算子不可用。因此不需要安装 `ultralytics`，也不要直接在已经跑通 OCR 的
Paddle 环境中升级/降级 Torch。

先在当前环境查看真正的导入结果：

```bash
python -c 'import torch, torchvision; from torchvision.ops import batched_nms; print("torch", torch.__version__, "torchvision", torchvision.__version__, "torch_cuda", torch.version.cuda, "cuda", torch.cuda.is_available())'
```

如果这条命令失败，优先复用已经能运行 LocateAnything/训练代码的 PyTorch 环境。YG 集群
现有 UI5 文档使用下面这个环境；先检查路径确实存在：

```bash
ICON_PYTHON=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything/bin/python
test -x "${ICON_PYTHON}"

"${ICON_PYTHON}" -c 'import numpy, PIL, torch, torchvision; from torchvision.ops import batched_nms; print("torch", torch.__version__, "torchvision", torchvision.__version__, "torch_cuda", torch.version.cuda, "cuda", torch.cuda.is_available())'
```

如果该绝对路径在当前机器不存在，就把它替换为 `conda env list` 中实际 LocateAnything/Torch
环境的 `bin/python`。如果路径存在但验证仍失败，先保留完整 traceback 和版本信息；不要盲目
执行 `pip install -U torch torchvision`，否则可能改坏正式训练环境。

验证成功后，给审计命令增加：

```bash
--icon-python "${ICON_PYTHON}"
```

也可以 `export ICON_PYTHON=/absolute/path/to/.../bin/python`，脚本会自动读取。主脚本和 text
worker 仍使用当前 Paddle Python，只有 icon worker 使用该 PyTorch Python。启动四个 icon
worker 前，脚本会先单进程检查 NumPy/Pillow、Torch、`torchvision.ops.batched_nms`、CUDA
可见性和 `model.pt` 是否能加载；失败时会打印未被 parser 隐藏的原始 traceback。

本次如果 text 已经完成而 icon 在 `0/17281` 失败，更新代码后仍可使用原来的
`--stage all --resume`。prepare 和全部 text shard 会校验后跳过，流水线直接回到 icon；不会
重新执行 17,281 张 OCR。也可把 `--stage all` 改为 `--stage icon`，icon 成功后再执行
`merge` 和 `crop-audit`。

## 4. 当前第二轮：只跑一条 crop-only 命令

17,281 张图片的 OCR、icon 和 merged detections 已经完整落盘。当前不要再用
`--stage all`，也不要重新执行 `prepare`、`text`、`icon` 或 `merge`。在
`Embodied-ui5-det-crop` 目录只执行：

```bash
bash shell/run_ui5_crop_audit.sh \
  --source-dir /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data \
  --locany-data-dir /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied/data/ui_defect_locany_v3 \
  --parser-root ../ui-region-parser \
  --output-dir work_dirs/ui5_crop_audit_20260825 \
  --gpus 0,1,2,3 \
  --workers-per-gpu 1 \
  --crop-workers 8 \
  --expected-unique-images 17281 \
  --crop-audit-name crop_audit_v3 \
  --stage crop-audit \
  --resume
```

这条命令只读取：

- `manifest/unique_images.jsonl`；
- `manifest/task_samples.jsonl`；
- `manifest/shards/shard_*.jsonl`；
- `detections/merged/detections.jsonl`；
- `detections/detector_config.json`。

它不会启动 Paddle、Torch 或 GPU worker。新结果写入 `crop_audit_v3/`，现有旧 audit 和整个
`detections/` 目录都保留原样。启动时会记录上述输入的文件 digest、image_id 数量及集合；结束
时再次核对，任何变化都会报错。

`--resume` 只跳过参数 digest 完全一致、JSONL 行数/image_id 检查通过、完成标记有效且落盘文件
存在的 shard。想比较另一组代码或参数时请换名字，例如
`--crop-audit-name crop_audit_v3_retry`，不要覆盖同名审计。

本轮 `crop_audit_v3` 已有有效的 geometry/materialization 完成标记。上述命令会复用
`TA_CTX015_H050` 的 geometry 和现有 47,930 个 region crop，不重新打开原图保存这些 PNG；
它只刷新统计、Excel、107 张失败四联图和训练门禁。终端最终应显示 OCR/icon 未运行，且
`detector_stages_executed=[]`。

只有在一个全新数据集还没有检测结果时，才需要从 `prepare → text → icon → merge` 逐阶段执行；
本轮不属于这种情况。

### 4.1 当前结果与统计分母

当前推荐候选是 `TA_CTX015_H050`。四个区域任务共 17,798 个正样本 GT bbox，其中
17,691 个被至少一个 crop 完整包含，GT bbox 完整覆盖率为
`17,691 / 17,798 = 99.3988%`。剩余 107 个失败 GT 包括 87 个
`partial_intersection` 和 20 个 `uncovered`。`ui_content_missing` 使用完整原图，覆盖率为
100%。

报告、Excel `summary` 和 `summary.json.metric_definitions` 对各指标使用以下统一口径：

- `gt_box_containment_recall`：被至少一个 crop 完整包含的 GT bbox 数 / 所有正样本中的 GT
  bbox 总数。负样本 `gt_count=0`，不进入分母；这不是成功图片数除以全部图片数。
- `positive_sample_success_rate`：一张正样本中全部 GT bbox 都被完整覆盖的正样本数 / 正样本
  总数。当前区域任务为 `11,150 / 11,250 = 99.1111%`；负样本不进入分母。
- `near_full_image_ratio`：union area / original area 大于 0.8 的图片样本数 / 该任务全部图片
  样本数，包含正样本和负样本。
- `gt_gain_over_1.25/1.5/2.0_ratio`：超过对应放大阈值的 GT bbox 数 / 已被 crop 完整包含且
  可以计算放大倍率的 GT bbox 数。
- `negative_samples` 只描述数据组成，不参与 GT 覆盖率或正样本全部成功率的分母。

107 个失败的固定分布如下；脚本会逐项校验，不一致直接停止，避免展示错 audit/config 的结果：

| 任务 | partial | uncovered | 合计 |
|---|---:|---:|---:|
| ui_occlusion | 27 | 19 | 46 |
| ui_cropping | 23 | 0 | 23 |
| ui_text_overflow | 2 | 1 | 3 |
| ui_text_ellipsis | 35 | 0 | 35 |
| ui_content_missing | 0 | 0 | 0 |
| 合计 | 87 | 20 | 107 |

| 页面密度 | partial | uncovered | 合计 |
|---|---:|---:|---:|
| sparse | 35 | 3 | 38 |
| medium | 49 | 17 | 66 |
| dense | 3 | 0 | 3 |
| 合计 | 87 | 20 | 107 |

### 4.2 四联图和人工失败归因

`--stage crop-audit --resume` 会自动调用 `scripts/render_ui5_crop_failures.py`。它只读取 manifest、
merged detections、`gt_failures.jsonl`、已完成的 task-aware geometry 和原图；不会导入或运行
PP-OCRv5/OmniParser。每个失败 GT 生成一张横向四联图：原图、OCR text detections、icon
detections、GT 与该任务 crop。正式 crop 图片不会被画框。

先打开可筛选总览：

```text
work_dirs/ui5_crop_audit_20260825/crop_audit_v3/failure_visualizations/gallery/index.html
```

`uncovered_all.html` 包含全部 20 个 uncovered；`representative_partial.html` 按任务和补偿量
small/medium/large 分桶展示接近 p50、p90 和最大值的代表样本。全部 107 条记录位于
`gt_failures_visualized.jsonl`。逐条查看四联图后，只编辑每行的 `manual_root_cause` 和
`manual_note`，原因必须是以下之一：

- `text_detector_miss`
- `icon_detector_miss`
- `gt_spans_multiple_components`
- `context_too_small`
- `task_linking_rule_mismatch`
- `annotation_suspect`
- `other`（必须在 `manual_note` 中说明）

完成全部 107 条人工判断后，用下面的纯汇总命令校验并刷新一页诊断报告；它不重新渲染 PNG：

```bash
python scripts/render_ui5_crop_failures.py \
  --output-dir work_dirs/ui5_crop_audit_20260825 \
  --crop-audit-name crop_audit_v3 \
  --config TA_CTX015_H050 \
  --expected-failures 107 \
  --expected-partial 87 \
  --expected-uncovered 20 \
  --summary-only \
  --require-manual-review
```

输出的 `failure_diagnosis_summary.json` 和 `gallery/diagnosis_summary.html` 按 task、density、
failure type、root cause 汇总，并展示每类原因最多 3 个代表案例。人工归因只用于判断是否值得
继续做下一轮几何规则，不能按单张 GT 坐标直接修 crop。本轮不新增 D/E/F 配置，也不重新落图。

## 5. 实时进度和剩余时间

默认每 10 秒汇总一次所有 GPU worker，在终端/集群日志打印：

```text
[进度 text 2/5] 12850/50000 images (25.7%) | 已耗时 00:41:23 | 速度 5.17 images/s | ETA 01:59:42
```

这里显示的是：当前阶段、流水线第几步、已完成/总数、百分比、当前实测吞吐、已耗时和当前
阶段预计剩余时间。刚启动、模型加载或还没有完成第一张图时，ETA 会显示 `--:--:--`；处理
一批图片后才会逐渐稳定。不同分辨率图片耗时不同，因此 ETA 是动态估计，不是承诺时间。

脚本还会原子更新：

```text
work_dirs/ui5_crop_audit_20260825/run_status.json
```

另开一个终端即可实时查看：

```bash
watch -n 5 'cat work_dirs/ui5_crop_audit_20260825/run_status.json'
```

默认参数通常不需要修改；若日志太密，可改为每 30 秒显示一次：

```bash
--progress-interval-seconds 30
```

本轮 `crop-audit` 分成两个独立 ETA：阶段 1 是五组 task-aware 候选的纯几何评价，阶段 2 是
最佳候选落图。使用 `--resume` 时，已验证完成的 geometry/materialized shard 会直接计入已完成
数量，ETA 只按本次剩余工作估算。阶段 1 的单位是 `image-candidates`，总数约为
`17,281 × 5 = 86,405`；这不表示图片被解码五遍。相同几何规则会在同一图片内跨候选复用。

### crop-audit 为什么分成两遍

旧实现对 17,281 张图的 A/B/C 三组重复解码原图并保存大量 PNG，实测 ETA 可接近 48 小时。
v3 实现为：

1. 阶段 1 使用 `--crop-workers 8` 个 CPU 进程，只读取 merged detections 做连通分量、合并、
   context、GT 离线评价、面积和坐标 round-trip；不打开原图、不写 PNG；
2. 每 500–1000 张沿用 manifest shard，原子写 `geometry/shard_*.jsonl` 和完成标记；
3. 同一图片内，候选间相同的几何规则只计算一次；四个区域任务分别生成 proposal，GT 不合并；
4. 按总体覆盖率、最低任务覆盖率、正样本成功率、像素减少和放大收益依次选择候选；
5. 阶段 2 每张图只解码一次；同一 bbox 跨任务只保存一个无框、无 mask 的 PNG；
6. `ui_content_missing` 直接引用完整原图 `[0,0,W,H]`，不再生成 whole PNG，也不重复归一化标签；
7. 当前推荐候选的全部 107 个失败 GT（20 个 uncovered、87 个 partial）逐条生成四联图；
   可选 overview 仍只用于其他异常类别抽样，不影响正式 crop。

五组候选含义：

| 候选 | occlusion/cropping | overflow/ellipsis | 作用 |
|---|---|---|---|
| C | H/V 0.025，context 0.20 | H/V 0.025，context 0.20 | 原配置 C 统一基线 |
| TA_CTX010_H035 | 上述基础上，最小图像 context 0.010 | H 0.035、V 0.025 | 元素任务补边，文字增强水平连接 |
| TA_CTX015_H035 | 最小图像 context 0.015 | H 0.035、V 0.025 | 比上一组更大最小补边 |
| TA_CTX010_H050 | 最小图像 context 0.010 | H 0.050、V 0.025 | 更强文字水平连接 |
| TA_CTX015_H050 | 最小图像 context 0.015 | H 0.050、V 0.025 | 两个增强方向的较强组合 |

元素任务的最小上下文为 `max(0.20 × component_size, ratio × image_size)`。所有规则只读取
检测框、图片尺寸和任务类型；GT 只用于离线统计和候选选择，不能逐样本修边或生成 fallback。

五个 `candidate_*/geometry/` 都会出现，只有 `materialized_candidate` 对应目录落正式 crop。
若至少一个候选通过严格 gate，`summary.json` 才会出现 `recommended_config` 并生成
`training_ready.json`；如果没有候选通过，只写 `best_candidate_config`，不会产生 training-ready
标记。旧 audit 不自动移动、不覆盖；目标名字已存在但参数不一致时直接报错，必须换
`--crop-audit-name`。

有效性报告按 text+icon 检测框总数分层：`sparse ≤ 50`、`medium = 51–150`、
`dense > 150`。每层分别输出 crop 数、union area、near-full、GT 放大收益和正负样本数；密集页
允许保留 near-full，不为了指标强拆。

CPU 充足时默认使用 8 个 worker；内存或共享存储压力较大可改为 4：

```bash
--crop-workers 4
```

抽样/异常 overview 数量可覆盖，但不建议全量渲染：

```bash
--overview-samples-per-task 50 \
--overview-anomalies-per-category 50
```

## 6. 五条阶段命令是什么意思？本轮需要逐条跑吗？

五条命令是给“从零开始的新数据集”或某个 GPU 阶段失败后单独恢复使用的。它们确实按
`prepare → text → icon → merge → crop-audit` 顺序逐条执行，但当前 17,281 张数据的前四步已经
完成，因此本轮不要运行下面这组命令：

```bash
COMMON_ARGS=(
  --source-dir /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data
  --locany-data-dir /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied/data/ui_defect_locany_v3
  --parser-root ../ui-region-parser
  --output-dir work_dirs/ui5_crop_audit_20260825
  --gpus 0,1,2,3
  --workers-per-gpu 1
  --crop-workers 8
  --expected-unique-images 17281
  --crop-audit-name crop_audit_v3
  --resume
)

bash shell/run_ui5_crop_audit.sh "${COMMON_ARGS[@]}" --stage prepare
bash shell/run_ui5_crop_audit.sh "${COMMON_ARGS[@]}" --stage text
bash shell/run_ui5_crop_audit.sh "${COMMON_ARGS[@]}" --stage icon
bash shell/run_ui5_crop_audit.sh "${COMMON_ARGS[@]}" --stage merge
bash shell/run_ui5_crop_audit.sh "${COMMON_ARGS[@]}" --stage crop-audit
```

只有处理全新数据时才从上到下执行。当前只使用第 4 节那条 `--stage crop-audit` 命令。
`--resume` 会验证 shard 行数、image_id 集合、输入/参数 digest 和完成标记；完整 shard 跳过，
不完整 shard 才重算。crop 参数变化始终使用新的 audit 名称并复用同一份 merged detections。

## 7. 1/2 processes per GPU benchmark（可选，不是正式流程）

正式运行默认保持 `--workers-per-gpu 1`。只有先用独立输出目录对同一批 2,000 张图实测，且
满足 GPU 利用率持续低于 40%、显存稳定低于 12 GB、2 processes/GPU 吞吐确实更高，才允许
改成 2。绝大多数情况下可以先跳过 benchmark，直接使用 1 process/GPU。

需要 benchmark 时，分别用 `--max-unique-images 2000` 建两个输出目录，再比较
`detections/text/stage_summary.json` 的 `throughput_images_per_second`。2 processes/GPU
还必须显式加 `--allow-two-processes-per-gpu`。benchmark 输出不能作为正式全量输出。

## 8. 输出目录

```text
work_dirs/ui5_crop_audit_20260825/
  run_status.json
  manifest/
    unique_images.jsonl
    task_samples.jsonl
    shards/shard_*.jsonl
    overlap/source_overlap.json
  detections/
    detector_config.json
    text/shard_*.jsonl
    icon/shard_*.jsonl
    merged/detections.jsonl
  crop_audit/                 # 旧审计，原样保留
  crop_audit_v3/
    audit_state.json
    candidate_C/
      geometry/shard_*.jsonl
      geometry/shard_*.done.json
      anomalies.json
    candidate_TA_CTX010_H035/
    candidate_TA_CTX015_H035/
    candidate_TA_CTX010_H050/
    candidate_TA_CTX015_H050/
    # 以下只存在于 materialized_candidate 对应目录
    candidate_<materialized>/
      materialized/shard_*.jsonl
      materialized/shard_*.done.json
      crops/
      overviews/
      preview/*.jsonl
    summary.json
    materialization_summary.json
    statistics.csv
    task_aware_manifest.jsonl
    gt_failures.jsonl
    gt_failures_visualized.jsonl
    failure_diagnosis_summary.json
    failure_visualizations/
      ui_occlusion/{partial_intersection,uncovered}/*.png
      ui_cropping/{partial_intersection,uncovered}/*.png
      ui_text_overflow/{partial_intersection,uncovered}/*.png
      ui_text_ellipsis/{partial_intersection,uncovered}/*.png
      gallery/{index,uncovered_all,representative_partial,diagnosis_summary}.html
    cross_task_supervision.jsonl
    training_ready.json        # 仅严格 gate 全部通过时存在
    ui5_crop_audit.xlsx
```

Excel 只包含五个有决策价值的 sheet：`summary`、`task_overlap`、`image_detail`、
`gt_failures`、`config_compare`。overview 和原始无框 crop 分开保存，并从 Excel 设置超链接。

## 9. 必须保持的原则

- 每张内容唯一的原图只检测一次；不同任务复用检测与 crop 图片，但 GT、prompt 和正负标签独立。
- basename 只用于冲突告警，不能作为图片身份。
- `ui_content_missing` 根据任务身份直接使用完整原图 `[0,0,W,H]`，原样复用输入的
  `gt_boxes_1000`；`label_transform_applied=false`。
- 正式 crop 是未画框、无 mask、未遮挡背景的原图矩形像素；扩张框只用于几何计算。
- dense 页面允许形成一个近整图连通区域，不按空白或长边二次强切。
- 部分相交 GT 对应的 crop 标记为 `training_eligible=false`，不能当负样本。
- 每个“原图 × 任务”最多保留一个完全不与 GT 相交的 hard negative。
- crop 参数变化只能读取 `detections/merged/detections.jsonl`，不能重新推理或覆盖检测结果。

## 10. 审计通过条件与后续训练

先提交审计报告，不启动训练。建议进入 full image、full+crop、full+crop 且推理使用 crop 三组
对照的最低条件：

- 四个局部任务总体 GT box 完整包含率不低于 99%；
- 每个局部任务不低于 98%；
- detector box 被 crop 边界切断数为 0；
- 区域 crop 坐标 round-trip error 大于 1 像素的数量为 0；
- partial crop 均为 `training_eligible=false`，每个原图 × 任务最多一个 hard negative；
- `ui_content_missing` 完整覆盖率为 100%，normalized GT 与输入完全一致；
- train/val 同内容重叠数为 0。

`next_stage_gate.conditions` 会显式记录原有六项和以下五项：
`same_content_cross_train_val_count_zero`、`content_missing_recall_equals_1`、
`content_missing_normalized_gt_mismatch_count_zero`、`input_snapshot_unchanged`、
`all_reports_written_successfully`。最终 `passes` 是全部 11 项的逻辑与。

每次刷新报告一开始都会使旧 `training_ready.json` 失效。只有 geometry/materialization、
content_missing、输入快照、JSONL/CSV/Excel/四联图报告全部成功且 gate 全部通过后，脚本才在
最后一步原子写入新 marker。marker 包含 audit state、输入快照和最终 summary 的 digest；运行中断
或任一检查失败都不会留下训练入口可以接受的旧 marker。

使用检测 crop 的训练入口会在启动前再次校验 marker 和三个 digest。显式设置 audit 目录：

```bash
export UI5_USE_DETECTION_CROPS=1
export UI5_CROP_AUDIT_DIR=/absolute/path/to/work_dirs/ui5_crop_audit_20260825/crop_audit_v3
bash shell/train_locany_ui_defect.sh ...
```

也可以单独执行同一项只读校验：

```bash
python scripts/validate_ui5_crop_training_ready.py \
  --audit-dir /absolute/path/to/work_dirs/ui5_crop_audit_20260825/crop_audit_v3
```

若未达到，先看 `gt_failures.jsonl` 和 overview，调整 link/context 参数；不得读取 GT 位置
直接补 crop。即使 gate 通过，也只是允许进入 full image、full+crop、full+crop 且推理使用 crop
三组对照，不能据此断言训练一定提升；本脚本始终输出 `training_started=false`，不会启动训练。

# UI5 检测 Crop 审计（CPT disabled）

本分支基于 `locany-cpt-v1@c06f1479a11b0175579994b880466b57bba50a87`。
本轮只复用已更新的 Detail Pyramid、relation/gate、BF16 修复和 UI5 训练/评测基础设施：

- 不加载 CPT 数据；
- 不开放 CPT 训练入口；
- 不启动正式训练；
- crop 覆盖率高只说明方案可用，不代表训练效果一定提升。

`ui-region-parser` 保持为同级独立工具仓库，固定在
`06eaebf8eb4ea01e61b690f2ff972bf614915918`。审计入口只复用它的 detector 类和
cropper 几何函数，不使用其基于 basename 的 annotation/image 聚合，也不会在全量检测时
保存 combined/stage 可视化。

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

## 4. 正式运行：最简单是一条命令

在 `Embodied-ui5-det-crop` 目录执行：

```bash
bash shell/run_ui5_crop_audit.sh \
  --source-dir /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data \
  --locany-data-dir /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied/data/ui_defect_locany_v3 \
  --parser-root ../ui-region-parser \
  --output-dir work_dirs/ui5_crop_audit_20260825 \
  --gpus 0,1,2,3 \
  --workers-per-gpu 1 \
  --stage all \
  --resume
```

`--stage all` 会按顺序自动完成：

1. `prepare`：读取五类 JSONL、检查图片、按内容去重、生成 manifest 和 shard；
2. `text`：四卡运行 PP-OCRv5，按 shard 落盘后退出 Paddle 进程；
3. `icon`：四卡运行 OmniParser icon detector，按 shard 落盘后退出 Torch 进程；
4. `merge`：检查数量、重复、缺失和尺寸，再生成唯一 `detections.jsonl`；
5. `crop-audit`：只用 CPU 读取落盘检测，运行 A/B/C crop、preview、统计、可视化和 Excel。

所以，正常首次运行不需要手动复制执行五条命令。

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

`prepare` 会显示源数据/训练数据重叠分析和 manifest 构建进度；`text`、`icon` 汇总四卡
worker；`merge` 显示合并数量；`crop-audit` 把 A/B/C 三组配置合并成一个总进度。使用
`--resume` 时，已验证完成的 shard 会直接计入已完成数量，ETA 只按本次剩余工作估算。

## 6. 什么时候才需要一条一条运行？

以下情况建议分阶段：集群任务有时限、需要在 OCR 后释放资源、某阶段失败后单独重跑，或想先
检查检测数量再裁图。所有命令使用同一个 `--output-dir`：

```bash
COMMON_ARGS=(
  --source-dir /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data
  --locany-data-dir /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied/data/ui_defect_locany_v3
  --parser-root ../ui-region-parser
  --output-dir work_dirs/ui5_crop_audit_20260825
  --gpus 0,1,2,3
  --workers-per-gpu 1
  --resume
)

bash shell/run_ui5_crop_audit.sh "${COMMON_ARGS[@]}" --stage prepare
bash shell/run_ui5_crop_audit.sh "${COMMON_ARGS[@]}" --stage text
bash shell/run_ui5_crop_audit.sh "${COMMON_ARGS[@]}" --stage icon
bash shell/run_ui5_crop_audit.sh "${COMMON_ARGS[@]}" --stage merge
bash shell/run_ui5_crop_audit.sh "${COMMON_ARGS[@]}" --stage crop-audit
```

这里确实是从上到下依次执行。`--resume` 会验证 shard 的行数、image_id 集合和完成标记；
完整 shard 会跳过，不完整 shard 才重跑。已经有 text/icon 检测结果时，反复调整 crop 参数只需
执行 `--stage crop-audit`，不会再调用 GPU 模型。

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
  crop_audit/
    config_A/
      crops/
      overviews/
      preview/*.jsonl
      anomalies.json
    config_B/
    config_C/
    summary.json
    statistics.csv
    task_aware_manifest.jsonl
    gt_failures.jsonl
    ui5_crop_audit.xlsx
```

Excel 只包含五个有决策价值的 sheet：`summary`、`task_overlap`、`image_detail`、
`gt_failures`、`config_compare`。overview 和原始无框 crop 分开保存，并从 Excel 设置超链接。

## 9. 必须保持的原则

- 每张内容唯一的原图只检测一次；不同任务复用检测与 crop 图片，但 GT、prompt 和正负标签独立。
- basename 只用于冲突告警，不能作为图片身份。
- `ui_content_missing` 根据任务身份使用一张近似整图，不依赖文件名，不读取 GT 决定 crop。
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
- `ui_content_missing` 单独按近整图策略统计。

若未达到，先看 `gt_failures.jsonl` 和 overview，调整 link/context 参数；不得读取 GT 位置
直接补 crop。

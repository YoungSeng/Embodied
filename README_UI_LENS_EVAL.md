# 用 M32 checkpoint 在 UI-Lens 上推理并计算 F1

本说明适用于 `codex/m32-cpt-sft-croponly-v1` 分支。
`scripts/prepare_ui_lens_eval.py` 是随本指南提交的独立转换器；在服务器对应分支执行
`git pull --ff-only` 即可同时获取脚本和本说明，无需手动复制文件。

已核对来源：

- [UI-Lens 数据卡](https://huggingface.co/datasets/wuhuohua/UI_lens)：单界面任务、`infos` 字段和像素 xywh 框。
- [实际文件目录](https://huggingface.co/datasets/wuhuohua/UI_lens/tree/main)：目前是 `label_for_single_UIs_cn`、`single_UIs_cn` 和 `Seq_UIs_cn`。
- [参考分支推理器](https://github.com/YoungSeng/Embodied/blob/codex/m32-cpt-sft-croponly-v1/scripts/inference_ui_defect_locany.py)。
- [参考分支评分器](https://github.com/YoungSeng/Embodied/blob/codex/m32-cpt-sft-croponly-v1/qwen3vl_merge_and_score_fixed_5tasks.py)。

本次先跑五个单界面任务。序列界面的文本不一致需要多图/历史信息输入，不能混入这次 UI5 结果。
下面使用整图推理得到第一版结果。`croponly` 的训练输入方式并不阻止整图推理，
但整图结果和 detector-scan 的结果代表不同推理设置，比较时应写清楚。

## 1. 进入服务器环境，设置目录

步骤 1–3 和下面的 3.5（下载、转换、统计、GT 可视化）在能访问这些路径的 CPU 开发机执行即可。
只有步骤 4 的模型推理和步骤 5 的全量推理需要已分配的 GPU；步骤 6 的 F1 评分只用 CPU。

```bash
conda activate /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything

export PROJECT=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-m32-cpt-sft-croponly-v1
export CKPT="$PROJECT/work_dirs/locany-ui5-m32-cpt3000-croponly-sourcebalanced-a800x4-v1/checkpoint-9000"
export BASE=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/hf_home/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0
export RAW=/mnt/bn/intelligent-service-yg/dataset/UI_lens
export DATA=/mnt/bn/intelligent-service-yg/dataset/UI_lens_ui5_eval_clip_v1
export OUT="$PROJECT/work_dirs/ui-lens-checkpoint9000-fullimage-clip-v1"

cd "$PROJECT"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
git branch --show-current
test -d "$CKPT"
test -d "$BASE"
python scripts/locany_ui5_checkpoint.py validate --checkpoint "$CKPT" --mode eval
```

确认分支为 `codex/m32-cpt-sft-croponly-v1`，校验返回 `valid: true`。
`BASE` 来自参考分支的环境示例，若本机不存在，替换成训练实际使用的 LocateAnything-3B 基座路径。
模型权重来自 `CKPT`；`BASE` 提供 processor/tokenizer 与兼容文件，不会用基座权重替换微调权重。

## 2. 开通访问并下载数据

先登录 [数据集页面](https://huggingface.co/datasets/wuhuohua/UI_lens)，接受其访问条件。
下载使用同一个有权限的 Hugging Face 账号的 read token；token 在登录提示中输入，不写到脚本。

```bash
python -c 'from huggingface_hub import login; login()'

python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="wuhuohua/UI_lens",
    repo_type="dataset",
    revision="da320f45691a727e5ca14abe52b6547ae42e9019",
    local_dir=os.environ["RAW"],
    allow_patterns=["README.md", "label_for_single_UIs_cn/*", "single_UIs_cn/*"],
    ignore_patterns=["*.DS_Store"],
    max_workers=4,
)
PY
```

固定版本是本次查询到的提交，便于复现。仅下载本次需要的单界面标注和图片。
下载中断可以重跑。[Hugging Face 下载文档](https://huggingface.co/docs/huggingface_hub/guides/download)

401/403 先检查账号权限和 token。如果当前环境没有 `huggingface_hub`，可在独立下载环境安装它，
下载到同一 `RAW` 目录后切回训练环境；无需升级 Torch、Transformers 等训练依赖。

## 3. 看真实标注，再转换

当前公开目录包含以下五个标注文件。正文需要权限：转换器依据公开数据卡实现，
尚未用受限的真实标注验证；先执行这段检查，确认实际字段及负样本编码。

```bash
python - <<'PY'
import json, os
from collections import Counter
from pathlib import Path

for p in sorted((Path(os.environ["RAW"]) / "label_for_single_UIs_cn").glob("*.jsonl")):
    rows = [json.loads(s) for s in p.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    print("\nFILE:", p.name, "ROWS:", len(rows))
    if rows:
        print("FIRST:", json.dumps(rows[0], ensure_ascii=False, indent=2))
    print("TARGETS:", Counter(str(r.get("infos", {}).get("target_problem")) for r in rows))
    print("EXPLICIT EMPTY BOX LISTS:", sum(r.get("infos", {}).get("box_list") == [] for r in rows))
PY
```

| 源文件 | 推理任务 | 评分器标签 |
|---|---|---|
| `container_overlap.jsonl` | `occlusion` | 元素重叠 |
| `cropped_content.jsonl` | `cropping` | 元素被裁切 |
| `text_overflow.jsonl` | `text_overflow` | 文字溢出容器 |
| `abnormal_text_ellipsis.jsonl` | `text_ellipsis` | 文字省略异常 |
| `undisplayed_content.jsonl` | `content_missing` | 内容未展示 |

将服务器工程更新到包含本指南的提交后执行：

```bash
# 只校验、输出统计，不写文件。
python scripts/prepare_ui_lens_eval.py --dataset-root "$RAW" --bbox-boundary-policy clip

# 校验通过后写入新的 DATA 目录。
python scripts/prepare_ui_lens_eval.py --dataset-root "$RAW" --output-dir "$DATA" --bbox-boundary-policy clip
cat "$DATA/conversion_summary.json"
```

转换默认会立即输出启动信息，并每 5 秒、每完成 100 条标注、每个任务开始/完成时打印进度。
日志包括 `stage`（当前阶段）、`task_rows`（该任务已完成/总数）、`total_rows`（全部已完成/总数和百分比）、
`elapsed`、`speed`、`eta_validation`（预计校验剩余时间）、已解码图片数、已处理的正负记录数、
裁剪框数和当前图片路径。开始时会先读五个标注文件确定真实总数；此时尚无 ETA，显示 `--`。

网络挂载目录读取大图较慢时，后台进度线程仍会周期打印当前操作。
如果数字不变，可以从 `current` 查看是哪张图片，并区分 `open_image`、`image_metadata`、
`decode_image`、`validate_boxes`、`write_jsonl` 等阶段。
ETA 按已完成标注的平均速度估算；首次解码图片通常比后续任务复用尺寸缓存更慢，因此估计值会变化。
`eta_validation` 只估算数据校验，不包含后续写文件时间；出现 100% 后以最终 `[UI-LENS:DONE]` 为完成标志。

希望更频繁输出时（以下是一整行命令）：

```bash
python -u scripts/prepare_ui_lens_eval.py --dataset-root "$RAW" --output-dir "$DATA" --bbox-boundary-policy clip --progress-interval-seconds 2 --progress-every 50
```

进度写到 stderr 并立即刷新，stdout 仍只输出最终 JSON 汇总，因此可独立保存两种输出：

```bash
python -u scripts/prepare_ui_lens_eval.py --dataset-root "$RAW" --output-dir "$DATA" --bbox-boundary-policy clip > conversion_result.json 2> conversion_progress.log
```

`--quiet` 可关闭进度。更新脚本不会改变已经运行中的旧进程；需要先停止旧进程，再更新并重新运行。
校验阶段中断后尚未写 `DATA`，可以直接重跑；若已开始写文件并留下输出目录，使用新的 `--output-dir`，
不要把不完整的目录用于推理。脚本会在读取图片之前检查输出目录是否已存在，避免等待很久后才发现目录冲突。

转换规则：

- `infos.image_path` 转为真实存在的绝对图片路径，写入 `images`。
- `infos.box_list` 的 `[x,y,w,h]` 转成 `[x,y,x+w,y+h]`，保持原图像素坐标，不乘除 1000。
  本指南显式选用 `--bbox-boundary-policy clip`：有部分落在图内的越界框取与图像的交集。
  不丢弃框、不改变样本正负标签，原始标注和裁剪记录一起保留。
- `answer.bbox` 保存框，`answer.types` 保存对应任务的中文标签。
- 各任务只使用其源文件已有的样本。明确的空框列表 `[]` 保留为负样本；不把未标注图片补成负样本。
- `image_size` 支持 `[W,H]` 和真实标注中出现的 `[[W,H]]`，去掉单层包装后核对实际尺寸。
  原始字段保留在元信息中；尺寸不一致、零/负面积框、完全位于图外的框、未知任务、缺失标注、任务内重复图和同名冲突仍会报错。
- `target_problem` 和源文件确定任务，原始 `label_list` 保留在元信息中。

例如源框 `[10,20,30,40]` 变为：

```json
{
  "images": ["/mnt/bn/intelligent-service-yg/dataset/UI_lens/single_UIs_cn/1.png"],
  "answer": {"bbox": [[10,20,40,60]], "types": ["元素被裁切"]}
}
```

对应任务的负样本为 `"answer": {"bbox": [], "types": []}`。
`messages` 不必生成，推理脚本按 `--tasks` 使用与训练一致的固定 prompt，评分器读取 `answer`。

若校验报错，先查看异常指向的源文件及行号，再按真实语义调整转换器。
尤其不能把 `null`、缺字段、零面积占位框直接当作已确认负样本。
如果某类源数据没有负样本，保留这一事实并在报告说明 image F1 的测试范围，不能凭空补负样本。
也不要强制使用旧 UI5 的 1555 张计数；使用本次转换统计。

真实标注中已遇到：图像 `3618 × 7866`，框 `[2778,4140,849,348]`。
其原始 xyxy 为 `[2778,4140,3627,4488]`，右边缘超出 9 像素；
`clip` 模式保存为 `[2778,4140,3618,4488]`。
默认不传参数时仍是 `--bbox-boundary-policy error`，严格拒绝越界，便于保留原来的检查行为。

`conversion_summary.json` 会记录 `bbox_boundary_policy`、各任务 `clipped_images`、`clipped_boxes`、
`max_clip_pixels` 和 `max_removed_area_ratio`；顶层 `clipped_image_task_records` 与 `clipped_boxes`
汇总受影响的记录数和框数。`bbox_clipping_audit.jsonl` 记录每条受影响记录的来源行号、原始框、裁剪后框、
最大边缘裁剪像素数和移除面积比例。每条转换后的样本也在 `extra_info.bbox_conversion` 中保存这些信息。
原始数据文件始终不变。正式报告中应说明 GT 使用了边界裁剪策略；修改该策略时使用新的 `DATA` 和 `OUT`。

## 3.5. 打印全量统计，并可视化转换后的 GT 框（仅 CPU）

在转换成功后执行。已经生成 `DATA` 的用户无需重新转换，直接运行这个检查脚本：

```bash
python scripts/inspect_ui_lens_eval.py \
  --input-dir "$DATA" \
  --output-dir "$DATA/inspection" \
  --samples-per-task 20 --seed 42
```

终端会打印每类的样本数、正样本数、负样本数、正样本比例、GT 框数、单图最大框数、
发生边界裁剪的图片记录数和框数、坐标转换不一致数，以及五类合计和去重图片数。
这些数字基于**全部五个 JSONL**，不是抽样数量。
`TOTAL(image-task)` 是“图片 × 任务”的记录数；去重图片数按解析后的图片路径计算。
同一图片可以在 cropping 是正样本、在 occlusion 是负样本，不能将五类记录数当成独立图片数。

输出内容：

```text
UI_lens_ui5_eval_clip_v1/inspection/
├── index.html                 # 离线可打开的图集，可筛任务和正负样本
├── dataset_stats.json         # 全量统计与抽样参数
├── dataset_stats.csv          # 每任务及合计统计表
├── samples.json               # 预览图片对应的路径、JSONL 行号、原始/转换后坐标
├── cropping/*.png
├── occlusion/*.png
└── ...                        # 其他三类
```

将整个 `inspection` 目录复制到本机后打开 `index.html`，或直接查看各任务 PNG。
图中左侧为原图，右侧框**直接取自转换后保存的 `answer.bbox`**，不重新转换后再画。
绿色为未裁剪框，橙色为经过边界裁剪的框；图集可以筛选“有裁剪”，坐标表会展示每个框的裁剪量。
每个框显示编号，HTML 中展开“查看原始与转换后的坐标”即可核对
原始 `[x,y,w,h]` 与转换后 `[x1,y1,x2,y2]`。点击预览图可查看原分辨率 PNG。
负样本显示 `NEGATIVE | boxes=0`，保留未画框的原图。

默认每类最多 20 张，尽量各选 10 张正负样本；某一类样本不足时用另一类补足。
发现坐标转换不一致的记录时最先展示，再优先展示发生边界裁剪的记录，剩余名额才做正负抽样。
因此当异常/裁剪样本较多时，预览中的正负数量不一定均衡。检查会独立核对 `error` 或 `clip` 策略、
原始 xywh、实际保存的 xyxy 和裁剪记录，合法且有记录的 clip 不计为转换错误。
真正的转换不一致会在统计里计数；报告仍会生成，命令返回非零退出码，
需先查看并解决问题再做推理。检查输出目录必须是新的，重复运行时换成 `inspection-v2` 等目录名。

只打印全量统计，不生成图片或文件：

```bash
python scripts/inspect_ui_lens_eval.py --input-dir "$DATA" --stats-only
```

查看全部标注图片时将 `--samples-per-task` 改为 `0`，并换一个新的输出目录。
全量可视化会生成更多 PNG，通常先抽查即可。

如果转换时遇到 `infos.image_size [[1206, 2622]] != actual size [1206, 2622]`，
先 `git pull --ff-only` 获取支持嵌套尺寸的修复，再重跑步骤 3。
这个尺寸错误发生在写输出前，所以通常还没有 `conversion_summary.json`；此时不要先运行 `cat` 或检查脚本。
如果遇到越界框错误，更新代码后使用上述 `--bbox-boundary-policy clip` 命令，并查看裁剪统计和橙色框。

## 4. 准备 checkpoint，先推理少量图片

此时才需要 GPU。建议先申请一张交互式 GPU，执行下面的每类两张图片检查；
通过后再选择在已有 GPU 配额内直接运行步骤 5，或提交正式 GPU 评测任务。
正式任务的启动命令使用步骤 5 的推理入口，不需要重新训练，也不要用训练提交入口来代替评测。
若平台给交互式会话的时限较短，全量推理建议提交正式任务。

参考分支的常规评测也会做 checkpoint patch，补齐自定义模型代码、配置和 processor 文件，
并验证 Relation/PBD 权重。使用同一个训练分支执行：

```bash
python scripts/patch_locany_checkpoint.py \
  --checkpoint "$CKPT" --base-model "$BASE" \
  --project-root "$PROJECT" --force --validate-relation-weights

python scripts/inference_ui_defect_locany.py \
  --checkpoint "$CKPT" --processor-path "$BASE" \
  --input-dir "$DATA" --output-dir "${OUT}-smoke" \
  --cuda-visible-devices 0 --device cuda:0 \
  --attn-implementation sdpa --vision-attn-implementation flash_attention_2 \
  --generation-mode hybrid --relation-gate-mode observe --enable-pbd \
  --inference-crop-mode full_image \
  --tasks all --max-images-per-task 2 \
  --save-raw-answer --save-visualization --fail-fast
```

检查 `${OUT}-smoke/_summary.json`，以及每类 `raw/`、`visualizations/`。
确认加载成功、图片没有缺失、框的坐标正常、无推理错误。
这只验证流程，不能用 smoke 预测对全量 GT 算正式 F1。
若缺少 FlashAttention 2，应先核对是否激活了原训练环境；上述多卡入口固定使用这个视觉后端。

## 5. 四卡跑五类全量推理

```bash
python scripts/run_ui5_parallel_inference.py \
  --checkpoint "$CKPT" --processor-path "$BASE" \
  --input-dir "$DATA" --output-dir "$OUT/predictions" \
  --gpu-devices 0,1,2,3 --attn-implementation sdpa \
  --inference-script "$PROJECT/scripts/inference_ui_defect_locany.py" \
  --relation-gate-mode observe --enable-pbd \
  --inference-crop-mode full_image --save-raw-answer
```

五类任务分配到四张卡独立推理；只有一张卡时将 `--gpu-devices` 改为 `0`。
进程中断后用相同参数重跑，入口支持按图片续推。
数据、checkpoint 或推理方式变化时使用新 `OUT`，防止复用旧结果。
本方案不传 `--expected-images-per-task 1555`，不使用旧 UI5 的 detector cache。

## 6. 复用原评分器计算 image F1 / bbox F1

仅在全量推理成功完成后执行：

```bash
python qwen3vl_merge_and_score_fixed_5tasks.py \
  --all_tasks --input_mode yolo_dir \
  --gt_dir "$DATA" --pred_root "$OUT/predictions" \
  --output_root "$OUT/evaluation" --run_name iou010 \
  --yolo_bbox_format xyxy --iou_thresh 0.1

cat "$OUT/evaluation/iou010/all_tasks_evaluation.txt"
```

这里 `yolo_dir` 指 LocateAnything 已输出的兼容 JSON 格式，不需要额外运行 YOLO。
不要用 Qwen 归一化坐标的 `dir` 解析方式，否则可能错误地再次缩放坐标。
结果同时有 `all_tasks_evaluation.json`，包含五类的 image / bbox precision、recall、F1。
JSON 的 `macro` 是各任务比例指标的算术平均，对应文本报告中的“五类平均”。
原分支没有直接输出 micro F1；如需它，可先合并五类 TP/FP/FN 后再计算，不能直接平均五类 F1。

- image F1：按任务判断该图是否有至少一个缺陷框，与框是否定位准确无关。
- bbox F1：每张图内对 GT 和预测框做 Hungarian 匹配，再以 IoU ≥ 0.1 计 TP，与原分支口径一致。
- 两者均为 `2TP / (2TP + FP + FN)`。五类的汇总 image 指标以“图片 × 任务”为单位，并非去重后的“任意一种缺陷”指标。
- 检查每类 `total_samples` 与转换器 `rows` 一致，`invalid_pred` 应为 0；评分日志的 `missing_files`、`parse_errors` 应为 0。
  旧评分器会惩罚缺失/非法预测，不能将未跑完的结果当作正式分数。

需要更严格的定位指标时，对同一批预测再算 IoU 0.5，无需重新推理：

```bash
python qwen3vl_merge_and_score_fixed_5tasks.py \
  --all_tasks --input_mode yolo_dir \
  --gt_dir "$DATA" --pred_root "$OUT/predictions" \
  --output_root "$OUT/evaluation" --run_name iou050 \
  --yolo_bbox_format xyxy --iou_thresh 0.5
```

评分输出目录必须是新的；重新评分时换一个 `--run_name`。

## 后续对齐 crop-only 训练的推理方式

如果需要与训练期间的 detector-scan 评测严格对照，应先读取原运行保存的 evaluation 元数据，
确认 crop mode、Gate mode、PBD 和切图参数，再为 UI-Lens 生成独立的 OCR/icon detector cache。
参考分支已有 `scripts/prepare_ui5_eval_detector_crops.py`，但还需要实际 parser/model 路径，
并按 UI-Lens 的真实图数验证缓存。不能套用原 UI5 的缓存和 1555 张约束。
推理 crops 必须只由图像/detector 决定，不能用 UI-Lens GT 框选 crop；最终预测应回映射到原图后评分。

不依赖 OCR/icon 的另一个已有选项是 `--inference-crop-mode lossless_tiling`，
但它与 detector-scan 是不同方法。改变推理方式时使用单独的输出目录并单独报告。

本结果是沿用当前工程评分器的 UI-Lens 单界面评测，不能自动等同于 UI-Lens 论文官方指标。
如果后续将其作为独立外部测试集，应检查这些图片是否出现在本次 CPT/SFT 的来源中。

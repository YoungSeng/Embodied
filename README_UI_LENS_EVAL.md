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

在已分配 GPU 的 Linux 节点执行；本地 Windows 工作区无法检查这个 `/mnt/bn` checkpoint。

```bash
conda activate /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything

export PROJECT=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-m32-cpt-sft-croponly-v1
export CKPT="$PROJECT/work_dirs/locany-ui5-m32-cpt3000-croponly-sourcebalanced-a800x4-v1/checkpoint-9000"
export BASE=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/hf_home/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0
export RAW=/mnt/bn/intelligent-service-yg/dataset/UI_lens
export DATA=/mnt/bn/intelligent-service-yg/dataset/UI_lens_ui5_eval_v1
export OUT="$PROJECT/work_dirs/ui-lens-checkpoint9000-fullimage-v1"

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
python scripts/prepare_ui_lens_eval.py --dataset-root "$RAW"

# 校验通过后写入新的 DATA 目录。
python scripts/prepare_ui_lens_eval.py --dataset-root "$RAW" --output-dir "$DATA"
cat "$DATA/conversion_summary.json"
```

转换规则：

- `infos.image_path` 转为真实存在的绝对图片路径，写入 `images`。
- `infos.box_list` 的 `[x,y,w,h]` 转成 `[x,y,x+w,y+h]`，保持原图像素坐标，不乘除 1000。
- `answer.bbox` 保存框，`answer.types` 保存对应任务的中文标签。
- 各任务只使用其源文件已有的样本。明确的空框列表 `[]` 保留为负样本；不把未标注图片补成负样本。
- 逐张核对 `image_size` 和实际尺寸，拒绝越界框、未知任务、缺失标注、任务内重复图和同名冲突。
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

## 4. 准备 checkpoint，先推理少量图片

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

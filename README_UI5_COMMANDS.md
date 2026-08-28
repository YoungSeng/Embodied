# LocateAnything UI5 常用命令

## 测试集水平 detector scan：先离线缓存，再只读评测

四个区域任务共享同一套 GT-free 水平切图，`ui_content_missing` 单独使用完整原图。区域切图
统一采用半开区间 `[y1,y2)`：相邻 crop 必须满足 `left.y2 == right.y1`，面积之和严格等于
原图面积。内部 seam 的 y 坐标只能精确等于某个原始 text/icon detector bbox 的 `y1` 或
`y2`；若某个 raw edge 落在任意其他 bbox 的严格内部，则先全局剔除。禁止使用 guard、合并
protected band、detector-free gap 中点或理想等分点生成 seam。安全 raw edge 不足时减少 crop
数量，最差保留一张完整原图，绝不穿框、扩张或重叠。测试 GT、训练侧 crop 和
`manual_gt_repair` 均不参与。

PP-OCRv5 与 icon detector 使用两个固定环境，不要互相安装依赖：

```bash
LA_PY=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything/bin/python
TEXT_PY=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/UI5PaddleOCR/bin/python
```

先扩大到每任务 200 张，再从实际 detector box 数的 sparse/medium/dense 三档各选最多 20 张：

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop

LA_PY=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything/bin/python
TEXT_PY=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/UI5PaddleOCR/bin/python

PYTHON_BIN="${LA_PY}" bash shell/run_ui5_eval_detector_preview.sh \
  --input-dir /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data \
  --parser-root ../ui-region-parser \
  --output-dir work_dirs/ui5_eval_detector_preview_v5 \
  --gpus 0,1,2,3 \
  --workers-per-gpu 1 \
  --text-python "${TEXT_PY}" \
  --icon-python "${LA_PY}" \
  --max-images-per-task 200 \
  --visualization-samples 60 \
  --scan-name horizontal_scan_v5_raw_detector_edge_aligned \
  --scan-context-pixels 0 \
  --target-guard-ratio 0 \
  --target-guard-min-pixels 0 \
  --target-guard-max-pixels 0 \
  --seam-edge-reference raw-detector-bbox \
  --seam-candidates safe-raw-detector-edges-only \
  --strict-vertical-partition \
  --resume
```

该命令内部按 `prepare → text → icon → merge → crop` 自动顺序执行，不需要手动跑五条命令。
text 阶段显式使用 `TEXT_PY`，icon 阶段显式使用 `LA_PY`；两个 preflight 都在 GPU worker
启动前完成。OCR 与 icon 不会同时常驻同一 GPU，`--resume` 会跳过已完成 shard。查看：

```text
work_dirs/ui5_eval_detector_preview_v5/horizontal_scan_v5_raw_detector_edge_aligned/gallery/index.html
work_dirs/ui5_eval_detector_preview_v5/horizontal_scan_v5_raw_detector_edge_aligned/summary.json
work_dirs/ui5_eval_detector_preview_v5/horizontal_scan_v5_raw_detector_edge_aligned/statistics.csv
work_dirs/ui5_eval_detector_preview_v5/horizontal_scan_v5_raw_detector_edge_aligned/preview_crops/
work_dirs/ui5_eval_detector_preview_v5/horizontal_scan_v5_raw_detector_edge_aligned/v4_v5_coordinate_compare.csv
```

如果已有 20 张 raw detections，只换几何 namespace，不重跑 text/icon：

```bash
"${LA_PY}" scripts/prepare_ui5_eval_detector_crops.py \
  --stage crop \
  --input-dir /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data \
  --parser-root ../ui-region-parser \
  --output-dir work_dirs/ui5_eval_detector_preview_20260827 \
  --scan-name horizontal_scan_v5_raw_detector_edge_aligned \
  --scan-max-crops 10 \
  --scan-target-height 960 \
  --scan-context-pixels 0 \
  --target-guard-ratio 0 \
  --target-guard-min-pixels 0 \
  --target-guard-max-pixels 0 \
  --seam-edge-reference raw-detector-bbox \
  --seam-candidates safe-raw-detector-edges-only \
  --strict-vertical-partition \
  --cache-scope preview \
  --visualization-samples 20 \
  --resume
```

raw 结果始终位于 `detections/{text,icon,merged}/`，不同几何版本写入不同 `--scan-name`
目录，因此只改 CPU geometry 不会覆盖或重新运行 detector。

另开终端可实时查看当前阶段、完成数、速度和 ETA：

```bash
watch -n 5 'cat work_dirs/ui5_eval_detector_preview_v5/run_status.json'
```

确认 preview 后，必须在训练前完成全量 17,281 张内容唯一图片的离线 cache。训练中的
step-0/1000/2000 评测不会现场启动 PaddleOCR 或 icon worker：

```bash
EVAL_CACHE=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_eval_detector_cache_horizontal_v5

"${LA_PY}" scripts/prepare_ui5_eval_detector_crops.py \
  --stage all \
  --input-dir /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data \
  --parser-root ../ui-region-parser \
  --output-dir "${EVAL_CACHE}" \
  --gpus 0,1,2,3 \
  --workers-per-gpu 1 \
  --text-python "${TEXT_PY}" \
  --icon-python "${LA_PY}" \
  --icon-model ../ui-region-parser/weights/icon_detect_v3/model.pt \
  --scan-name horizontal_scan_v5_raw_detector_edge_aligned \
  --scan-max-crops 10 \
  --scan-target-height 960 \
  --scan-context-pixels 0 \
  --target-guard-ratio 0 \
  --target-guard-min-pixels 0 \
  --target-guard-max-pixels 0 \
  --seam-edge-reference raw-detector-bbox \
  --seam-candidates safe-raw-detector-edges-only \
  --strict-vertical-partition \
  --cache-scope full_test \
  --visualization-samples 60 \
  --resume

"${LA_PY}" scripts/validate_ui5_eval_detector_cache.py \
  --cache-dir "${EVAL_CACHE}" \
  --scan-name horizontal_scan_v5_raw_detector_edge_aligned \
  --cache-scope full_test \
  --expected-unique-images 17281 \
  --require-strict-nonoverlap \
  --require-raw-detector-edge-alignment \
  --require-detector-unique-containment \
  --require-ready
```

ready marker 最后原子生成并绑定输入 JSONL、内容集合、parser、detector 配置与运行时、shard、
merged detections、几何配置、scan manifest 和报告 digest。正式评测默认
`--eval-detector-cache-mode readonly`；缓存缺失或 digest 改变时 fail closed，不回退现场检测或全图。
schema-v5 硬门禁同时要求：overlap/gap/duplicate pixel 均为 0，tile 面积和、union 面积与原图
面积严格相等，processed pixel ratio=1，每个 detector bbox 唯一归属一张 crop，seam cross、
boundary cut、balanced fallback、full-in-multi、duplicate 和 nested 均为 0；并要求每条 seam
属于全局安全 raw bbox edge 集合、到最近原始 detector edge 的距离严格为 0。非零 guard 和
schema-v4 marker 会被拒绝；preview marker 也不能用于正式训练评测。

marker 中的 `cache_scope` 明确区分 `preview` 与 `full_test`。preview 必须记录正数
`max_images_per_task`；full-test 必须为 0，并绑定显式的 17,281 张预期内容唯一图片。训练/周期
评测默认要求 `full_test`，所以 20/200 张预览即使几何全部通过也会在 LocateAnything worker
启动前 fail closed。

全量 validator 通过后，用同一 cache 做只读评测 smoke：

```bash
"${LA_PY}" scripts/submit_locany_ui5.py \
  --machine a800 \
  --resource-group aiai_locate \
  --gpus 4 \
  --eval-checkpoint /path/to/checkpoint \
  --eval-step 1000 \
  --eval-inference-crop-mode detector_scan \
  --eval-detector-cache "${EVAL_CACHE}" \
  --eval-detector-cache-mode readonly \
  --eval-scan-name horizontal_scan_v5_raw_detector_edge_aligned \
  --require-cache-scope full_test \
  --require-strict-nonoverlap \
  --require-raw-detector-edge-alignment \
  --require-detector-unique-containment \
  --eval-max-images-per-task 20 \
  --run-name locany-ui5-detector-scan-readonly-smoke
```

日志必须包含 `detector cache: readonly validated`，且不能出现 PaddleOCR/icon worker 启动日志。

## v4.1 当前执行顺序

先完成 [README_UI5_CROP_AUDIT.md](README_UI5_CROP_AUDIT.md) 顶部的唯一一条
`shell/run_ui5_gt_repair.sh` 命令。只有新目录
`crop_audit_v4_gt_repair/training_ready.json` 生成且下面的独立校验通过，才允许跑 20-step
smoke；smoke checkpoint 可恢复后，才提交正式训练。不要一次把三条命令同时后台启动。

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop

UI5_AUDIT_DIR=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_crop_audit_20260825/crop_audit_v4_gt_repair
EVAL_CACHE=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_eval_detector_cache_horizontal_v5
UI5_CROP_META=${UI5_AUDIT_DIR}/training_recipes/ui_defect_5class_train_full_plus_crop.json

python scripts/validate_ui5_crop_training_ready.py \
  --audit-dir "${UI5_AUDIT_DIR}" \
  --recipe "${UI5_CROP_META}"
```

校验不仅看 marker 是否存在，还会重新比较 audit state、input snapshot、summary、排除清单、
recipe/JSONL/recipe summary 的 digest。任何一项变化都会在 torchrun 前拒绝启动，不能静默
回退到全图 META_PATH。

## v4.1 A800 四卡 20-step crop smoke

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop

UI5_AUDIT_DIR=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_crop_audit_20260825/crop_audit_v4_gt_repair

python scripts/run_locany_ui5_local_debug.py \
  --machine a800 \
  --gpus 4 \
  --cuda-devices 0,1,2,3 \
  --max-num-tokens 12800 \
  --max-steps 20 \
  --save-steps 20 \
  --use-detection-crops \
  --crop-audit-dir "${UI5_AUDIT_DIR}" \
  --crop-train-mode full_plus_crop \
  --run-name locany-ui5-v4-gtcrop-smoke20
```

训练启动日志必须显示：

- `UI5_USE_DETECTION_CROPS=1`；
- 最终 `META_PATH` 以 `ui_defect_5class_train_full_plus_crop.json` 结尾；
- full/crop/GT-repair/excluded 记录数；
- recipe 校验显示 `gt_repair_action_count=106` 且
  `gt_repair_action_mapped_count=106`；
- 至少一个 `manual_gt_repair` crop 路径和多个 `raw_detector` crop 路径；
- dataloader 平衡日志显示 `manual_gt_repair retention after balancing`，其中 record 与
  repair GT key 均为完整的 `X/X`；
- 四卡 `MAX_NUM_TOKENS=12800`、`GRADIENT_ACCUMULATION_STEPS=2`；
- 环境 preflight 通过。

完成后校验可恢复 checkpoint：

```bash
python scripts/locany_ui5_checkpoint.py validate \
  --checkpoint work_dirs/locany-ui5-v4-gtcrop-smoke20/checkpoint-20 \
  --mode resume \
  --expected-ranks 4
```

只有输出包含 `"valid": true`，且 `training_args.bin` 非空、DeepSpeed model/optimizer、
trainer state 和 4 个 dataloader rank state 均完整，才进入正式训练。环境 pre/post fingerprint
也必须一致。

## v4.1 A800 四卡正式 crop 训练

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop

UI5_AUDIT_DIR=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_crop_audit_20260825/crop_audit_v4_gt_repair
EVAL_CACHE=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_eval_detector_cache_horizontal_v5

python scripts/submit_locany_ui5.py \
  --machine a800 \
  --resource-group aiai_locate \
  --gpus 4 \
  --max-num-tokens 12800 \
  --install-system-runtime-deps \
  --use-detection-crops \
  --crop-audit-dir "${UI5_AUDIT_DIR}" \
  --crop-train-mode full_plus_crop \
  --enable-eval \
  --eval-at-start \
  --eval-interval-steps 1000 \
  --eval-inference-crop-mode detector_scan \
  --eval-detector-cache "${EVAL_CACHE}" \
  --eval-detector-cache-mode readonly \
  --eval-scan-name horizontal_scan_v5_raw_detector_edge_aligned \
  --require-cache-scope full_test \
  --require-strict-nonoverlap \
  --require-raw-detector-edge-alignment \
  --require-detector-unique-containment \
  --max-steps 16000 \
  --save-steps 4000 \
  --run-name locany-ui5-v4-gtcrop-a800x4
```

这条命令现在默认使用上述 `detector_scan` 评测。若要保留旧的全图或纯规则网格对照，显式
设置以下之一，并使用不同的 `--run-name`：

```bash
--eval-inference-crop-mode full_image
--eval-inference-crop-mode lossless_tiling
```

三种模式的局部预测都会先映射回原图再做跨 tile 去重；任何评测模式都禁止读取 GT repair。

如改为八卡，只改：

```text
--gpus 8 --max-num-tokens 25600
```

八卡固定 `GRADIENT_ACCUMULATION_STEPS=1`；crop recipe、marker、评测和 checkpoint 规则不变。

运行时依赖修复来源为 `de30357f73b6b393840efcb7eb3ca37182e86cc4`，checkpoint/SIGBUS
修复来源为 `ed5add660608b93ed4f9d5c68efb1c04478aa6bd`；crop 分支完成实现提交为
`a0ee63abbf7ec702a5891e4f9bb4bcd1b5a02dbb`，进度实现提交为
`b353b6d2c41a1751dab3bee8bb1a5cb7bd9ea727`。

---

## 以下为旧版全图训练和运行排障命令

以下内容保留用于历史对照。v4.1 crop 训练不要使用旧工程目录或下面不带
`--use-detection-crops` 的命令替代上面的 smoke/正式命令。

先进入工程并激活现有环境：

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied
conda activate /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything
```

## 本地四卡调试：跳过所有评测，直接训练 20 step

A100 使用 `a800` profile，即 SDPA、四卡 accumulation=2、`MAX_NUM_TOKENS=12800`：

```bash
python scripts/run_locany_ui5_local_debug.py \
  --machine a800 \
  --gpus 4 \
  --cuda-devices 0,1,2,3 \
  --max-num-tokens 12800 \
  --max-steps 20 \
  --run-name locany-ui5-v4-checkpoint-smoke20
```

该命令直接进入正式训练使用的同一条 pipeline，但固定关闭 step-0 和周期评测。输出写入当前工程的 `work_dirs/locany-ui5-v4-local-.../`。

调试命令默认在最后一个 step 保存完整 checkpoint。训练结束后必须验证它确实可恢复：

```bash
python scripts/locany_ui5_checkpoint.py validate \
  --checkpoint work_dirs/locany-ui5-v4-checkpoint-smoke20/checkpoint-20 \
  --mode resume --expected-ranks 4
```

只有返回 `"valid": true`，才提交长训练。这一步会检查非空的
`training_args.bin`、`trainer_state.json`、DeepSpeed optimizer/model state 和四个
dataloader rank state，而不只是检查模型权重。

如果模型或训练 JSON 不在默认路径：

```bash
python scripts/run_locany_ui5_local_debug.py \
  --machine a800 --gpus 4 --max-steps 20 \
  --base-model /path/to/LocateAnything-3B \
  --meta-path /path/to/ui_defect_5class_train.json
```

只检查最终命令和参数，不启动训练：

```bash
python scripts/run_locany_ui5_local_debug.py --gpus 4 --dry-run
```

## A800 四卡正式训练

正式任务默认先检查 `cv2` 和 LocateAnything `AutoProcessor`。如果当前任务容器缺少
`libGL.so.1`，会在容器内自动执行 `apt-get` 安装 `libgl1 libglib2.0-0`，安装成功并
通过复检后才启动四个推理 worker。无需修改共享 Conda 环境中的 OpenCV。

原资源组 `1602`（默认，不写 `--resource-group` 也一样）：

```bash
python scripts/submit_locany_ui5.py \
  --machine a800 \
  --resource-group default \
  --gpus 4 \
  --max-num-tokens 12800 \
  --install-system-runtime-deps \
  --enable-eval \
  --max-steps 16000 \
  --save-steps 4000 \
  --run-name locany-ui5-v4-imagegatefix-a800x4
```

新资源组 `ies_aiai_experience/AIAI_locate`（group `2146`）：

```bash
python scripts/submit_locany_ui5.py \
  --machine a800 \
  --resource-group aiai_locate \
  --gpus 4 \
  --max-num-tokens 12800 \
  --install-system-runtime-deps \
  --enable-eval \
  --max-steps 16000 \
  --save-steps 4000 \
  --run-name locany-ui5-v4-imagegatefix-a800x4
```

`aiai_locate` 会自动使用 group `2146` 和队列 `compute-3302-yg-cloudnative-ai-aiai.locate-guarantee`，其他训练参数及挂载路径不变。

## Torch 环境与中断 checkpoint

训练任务运行期间，禁止对它正在使用的共享 `ENV_DIR` 执行 `pip install/uninstall`、
`conda install` 或覆盖 Torch/DeepSpeed/MAGI。运行中的进程已经 mmap 了这些 `.so`；
若文件被原地替换或截断，后续访问尚未载入的页面可能直接收到 `SIGBUS`，不会产生
Python traceback。这与本次模型写完后、`training_args.bin` 为 0 字节的现象一致。

需要修环境时，应停止使用该环境的任务，或先克隆到新的版本化目录，再让新任务使用：

```bash
conda create --prefix \
  /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything-v4-fix1 \
  --clone \
  /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything
```

训练入口会在 torchrun 前后记录 Torch、Transformers、DeepSpeed、MAGI 以及 Torch
共享库的指纹到 `<output_dir>/environment/`。同一次运行内指纹变化会以 exit code 46
失败。每次 checkpoint 还会写 `checkpoint_save_trace.jsonl`；再次发生 native signal
时，最后一条 `START` 能精确指出停在模型、`training_args.bin`、optimizer 还是 RNG。

若 checkpoint 只有完整模型 shard，但 `training_args.bin` 为 0 或缺少 optimizer/
trainer/dataloader state，它仍可用 `--mode eval` 评测，但绝不能 resume。保留故障目录，
并用新的 `--run-name` 从头训练，例如：

```bash
python scripts/submit_locany_ui5.py \
  --machine a800 --resource-group aiai_locate \
  --gpus 4 --max-num-tokens 12800 --enable-eval \
  --max-steps 16000 --save-steps 4000 \
  --run-name locany-ui5-v4-imagegatefix-a800x4-retry1
```

如果希望依赖缺失时直接失败而不安装，可额外传入：

```bash
--no-install-system-runtime-deps
```

## 本地检查正式评测运行环境

先在本地开发机验证正式任务死亡的位置，不需要 GPU：

```bash
python scripts/preflight_locany_runtime.py \
  --processor-path /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/hf_home/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0
```

再用一张本地 GPU 只跑 cropping 的 1 张图片：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_ui_defect_locany.py \
  --checkpoint work_dirs/locany-ui5-v4-relationfix-a800x4/checkpoint-0 \
  --processor-path /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/hf_home/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0 \
  --input-dir /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data \
  --output-dir /tmp/locany-ui5-local-eval-smoke \
  --cuda-visible-devices 0 --device cuda:0 \
  --attn-implementation sdpa \
  --vision-attn-implementation flash_attention_2 \
  --generation-mode hybrid --tasks cropping --max-images-per-task 1 \
  --skip-figma --fail-fast \
  --relation-gate-mode observe --relation-gate-threshold 0.5
```

架构增加 image Gate 后不要继续使用旧的 `relationfix-a800x4` 输出目录；其中的
`checkpoint-1000/2000` 缺少 `relation_pyramid.image_gate_heads`，必须使用新的
`run-name` 从 checkpoint-0 开始。

## A800 八卡正式训练

```bash
python scripts/submit_locany_ui5.py \
  --machine a800 \
  --gpus 8 \
  --max-num-tokens 25600 \
  --install-system-runtime-deps \
  --enable-eval \
  --max-steps 16000 \
  --save-steps 4000 \
  --run-name locany-ui5-v4-relationfix-a800x8
```

## H20 四卡正式训练

```bash
python scripts/submit_locany_ui5.py \
  --machine h20 \
  --gpus 4 \
  --max-num-tokens 12800 \
  --install-system-runtime-deps \
  --enable-eval \
  --max-steps 16000 \
  --save-steps 4000 \
  --run-name locany-ui5-v4-relationfix-h20x4
```

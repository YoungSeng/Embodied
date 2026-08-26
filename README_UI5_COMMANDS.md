# LocateAnything UI5 常用命令

## v4.1 当前执行顺序

先完成 [README_UI5_CROP_AUDIT.md](README_UI5_CROP_AUDIT.md) 顶部的唯一一条
`shell/run_ui5_gt_repair.sh` 命令。只有新目录
`crop_audit_v4_gt_repair/training_ready.json` 生成且下面的独立校验通过，才允许跑 20-step
smoke；smoke checkpoint 可恢复后，才提交正式训练。不要一次把三条命令同时后台启动。

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop

UI5_AUDIT_DIR=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_crop_audit_20260825/crop_audit_v4_gt_repair
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
  --max-steps 16000 \
  --save-steps 4000 \
  --run-name locany-ui5-v4-gtcrop-a800x4
```

这条命令的评测默认仍是 full-image，适合作为 `full+crop` 训练对照。如果要跑第三组
“full+crop 训练，并在推理/评测时使用无遗漏重叠切图”，在同一命令中额外添加：

```bash
--eval-inference-crop-mode lossless_tiling
```

`lossless_tiling` 不读取 GT repair：1–10 个普通矩形 tile 的并集覆盖原图 100%，最后一行和
最后一列贴齐边界，局部预测先映射回原图再跨 tile 去重；`ui_content_missing` 始终保留完整
原图。必须使用不同 `--run-name` 区分两组评测。

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

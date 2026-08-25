# LocateAnything UI5 常用命令

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
  --max-steps 20
```

该命令直接进入正式训练使用的同一条 pipeline，但固定关闭 step-0 和周期评测。输出写入当前工程的 `work_dirs/locany-ui5-v4-local-.../`。

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

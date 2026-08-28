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
  --max-steps 20 \
  --run-name locany-ui5-v4-checkpoint-smoke20
```

TC-MSED 主模型 M3 的本地 20-step 调试使用同一入口，只增加架构阶段：

```bash
python scripts/run_locany_ui5_local_debug.py \
  --machine a800 --gpus 4 --cuda-devices 0,1,2,3 \
  --max-num-tokens 12800 --max-steps 20 \
  --tc-msed-stage m3 \
  --run-name locany-ui5-tcmsed-m3-a800x4-smoke20
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

## TC-MSED 分阶段实验

`--tc-msed-stage` 的含义是：`v4` 原基线、`m1` TCSR、`m2` 加 Hungarian
set localizer、`m3` 加动态 slot PBD 和 coordinate bridge、`m4` 加 soft Gate、
`m5` 再给 overlap 开 rank-8 adapter。每个阶段必须使用新的 `--run-name`，不能从
结构不同的旧 checkpoint resume。

Gate 模块并不是到 M4 才存在：M1–M3 仍训练 image Gate 和 slot Gate，slot Gate 也
参与 slot 路由；只是 image-level `p_defect` 在 `observe` 模式不改写最终生成。M4
新增的是“把 image Gate 作为一次初始 `<box>/none` soft prior 注入生成”，用于隔离
定位架构收益和 Gate 输出收益。M5 仅是 overlap rank-8 adapter 消融，不是默认主模型。

M4/M5 的周期评测会先对同一 checkpoint 跑一遍 `observe` 作为真正 raw 对照，再跑
soft Gate 结果；因此推理耗时约为 M1–M3 的两倍。Excel 的 `raw_*` 与 `soft_*` 来自
这两次独立生成，离线 threshold sweep 只标记为 diagnostic upper bound。

```bash
# M1 / M2 / M3 均先跑到 3000 step，逐阶段验收
python scripts/submit_locany_ui5.py \
  --machine a800 --resource-group aiai_locate \
  --gpus 4 --max-num-tokens 12800 --enable-eval \
  --tc-msed-stage m3 --max-steps 3000 --save-steps 3000 \
  --run-name locany-ui5-tcmsed-m3-a800x4
```

同一 checkpoint 的 PBD on/off 诊断不关闭 Relation/slots；在已经分配 GPU 的节点运行：

```bash
python scripts/run_ui5_eval.py \
  --checkpoint /path/to/checkpoint-3000 --base-model /path/to/LocateAnything-3B \
  --step 3000 --machine-type a800 --gpu-count 4 --max-num-tokens 12800 \
  --eval-gpu-devices 0,1,2,3 --attn-implementation sdpa \
  --input-dir /path/to/data --output-dir /tmp/ui5-pbd-on \
  --scorer-root "$PWD" --project-root "$PWD" --enable-pbd

python scripts/run_ui5_eval.py \
  --checkpoint /path/to/checkpoint-3000 --base-model /path/to/LocateAnything-3B \
  --step 3000 --machine-type a800 --gpu-count 4 --max-num-tokens 12800 \
  --eval-gpu-devices 0,1,2,3 --attn-implementation sdpa \
  --input-dir /path/to/data --output-dir /tmp/ui5-pbd-off \
  --scorer-root "$PWD" --project-root "$PWD" --no-enable-pbd
```

checkpoint-3000 评测完成后生成每任务 10 张 GT/coarse slot/最终框/slot 绑定图：

```bash
python scripts/render_ui5_slot_diagnostics.py \
  --prediction-dir work_dirs/locany-ui5-tcmsed-m3-a800x4/inference-checkpoint-3000-full \
  --gt-dir /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data \
  --scorer-root "$PWD" \
  --output-dir work_dirs/locany-ui5-tcmsed-m3-a800x4/evaluation/slot-diagnostics-3000 \
  --per-task 10
```

绿色为 GT，橙色为 coarse slot，红色为最终框；标签中的 `/sN` 是该生成框绑定的
relation slot。脚本同时生成 `slot_diagnostics_manifest.json`。

正式四卡仍固定 accumulation=2/12800；八卡固定 accumulation=1/25600。除这三项
GPU 相关配置外，M1–M5 共用完全相同的数据、prompt、优化器和评测 pipeline。

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

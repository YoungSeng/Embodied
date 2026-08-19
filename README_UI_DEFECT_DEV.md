# LocateAnything UI 缺陷检测：外网开发与内网训练

> LocateAnything UI5 v4 的统一 A800/H20、4/8 卡训练与自动周期评测入口见
> [`README_LOCANY_UI5_PIPELINE.md`](README_LOCANY_UI5_PIPELINE.md)。新任务请使用
> `python scripts/submit_locany_ui5.py ...`；本文件中的旧机器专用 YAML 命令仅保留作历史参考。

这套文件用于把同一份 LocateAnything UI 缺陷代码分成两个运行档位：

- 外网单张 RTX 4090 只做冒烟测试，确认模型加载、样例读取、forward、backward、optimizer step 和日志保存都能运行。
- 内网 H20/A800 使用真实 v3 数据和正式超参数，执行完整训练、推理与评测。

外网和内网调用同一个 `eaglevl/train/locany_finetune_magi_stream.py`。差异只集中在 `shell/run_locany_ui_defect.sh` 的 profile 中，避免维护两套启动代码。

## 文件放置

将本压缩包中的内容复制到 `Eagle/Embodied` 根目录。复制后应有：

```text
Embodied/
├── eaglevl/
├── deepspeed_configs/
├── samples/ui_defect_locany_smoke_real/
├── scripts/generate_ui_defect_locany_smoke.py
├── scripts/validate_ui_defect_locany_sample.py
├── scripts/export_ui_defect_locany_smoke.py
├── shell/run_locany_ui_defect.sh
├── shell/train_locany_ui_defect.sh
└── jobs/locany_ui5_v3_a800x8_merlin.yaml
```

当前 4090 profile 默认使用 `samples/ui_defect_locany_smoke_real` 中的 10 条已导出样例：五类缺陷各一条正样本和一条负样本。它们只用于验证训练管线，不用于报告指标。

## 三个运行档位

| profile | GPU | attention | 单条/packed token 上限 | 更新范围 | 默认步数 | 默认数据 |
| --- | --- | --- | --- | --- | ---: | --- |
| `4090` | 1× RTX 4090 | SDPA | 4096 / 4096 | 冻结 LLM 和视觉塔，仅训练 MLP | 2 | 10 条合成样例 |
| `h20` | 4× H20 | Magi | 8192 / 25600 | 全参数 | 25000 | 内网 v3 |
| `a800` | 8× A800 | SDPA | 8192 / 25600 | 全参数 | 25000 | 内网 v3 |

正式档位按统一模型要求使用 8192/25600 配置。官方上游文档把非 Hopper 的 SDPA 描述为约 4K 的短序列路径；如果 A800 出现 OOM 或 attention mask 问题，可先用下列配置诊断（不能作为正式 8192 实验结果）：

```bash
MAX_SEQ_LENGTH=4096 \
MAX_NUM_TOKENS_PER_SAMPLE=4096 \
MAX_NUM_TOKENS=4096 \
bash shell/run_locany_ui_defect.sh a800
```

所有 profile 参数都可以用同名环境变量覆盖。profile 只提供默认值，不会盖掉你显式传入的超参数。

## 外网 4090：第一次启动

以下命令在 Linux、NVIDIA 驱动和 CUDA 已可用的前提下执行。官方仓库当前固定的主要依赖包括 `transformers==4.57.1`、`deepspeed==0.15.4`、`accelerate==1.5.2` 和 `tokenizers==0.22.0`。

```bash
git clone https://github.com/NVlabs/Eagle.git eagle
cd eagle/Embodied

# 把本包内容复制到这里，再创建环境
conda create -n locateanything python=3.10 -y
conda activate locateanything
pip install -e .

# 模型权重不要提交 Git
hf download nvidia/LocateAnything-3B \
  --local-dir models/LocateAnything-3B

python scripts/validate_ui_defect_locany_sample.py
bash shell/run_locany_ui_defect.sh 4090
```

成功标准不是 loss 数值，而是日志中至少完成 2 个 optimizer steps，最后出现：

```text
TRAIN_STATUS: SUCCESS
TRAIN_EXIT_CODE: 0
```

默认输出到 `work_dirs/locany-3b-ui5-4090-smoke-*`。如果 24 GB 显存仍然不足，先保持样例图不变，只降低 token 预算：

```bash
MAX_SEQ_LENGTH=3072 \
MAX_NUM_TOKENS_PER_SAMPLE=3072 \
MAX_NUM_TOKENS=3072 \
bash shell/run_locany_ui_defect.sh 4090
```

不要用这个 profile 的 loss、速度或 checkpoint 做实验结论。它冻结了大部分参数，目的只是尽快暴露代码、数据格式、模型结构和依赖问题。

## 内网 H20：正式训练

代码在外网修改并提交后，内网只需要拉取代码。真实数据、模型权重和输出仍留在内网挂载盘：

```bash
cd /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Eagle/Embodied
git pull

bash shell/run_locany_ui_defect.sh h20
```

等价的显式写法如下，适合临时改实验名或步数：

```bash
ROOT_PATH=/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace \
DATA_VERSION=v3 \
VERSION=v3_h20x4 \
MAX_STEPS=25000 \
RUN_NAME=locany-3b-ui5-h20-full-v3_h20x4-en \
bash shell/run_locany_ui_defect.sh h20
```

H20 profile 会检查 `magi_attention` 是否已经安装。正式运行前应安装与你当前环境一致的 MagiAttention 1.0.5。

## 内网 A800 / Merlin：正式训练

交互式启动：

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied
git pull
bash shell/run_locany_ui_defect.sh a800
```

Merlin 使用 `jobs/locany_ui5_v3_a800x8_merlin.yaml`。该 YAML 已补齐 `PROJECT_ROOT`、`OUTPUT_BASE`、`HF_HOME` 和本地模型路径，并把训练参数交给同一个 profile 管理：

```bash
mlx job submitv2 --path jobs/locany_ui5_v3_a800x8_merlin.yaml
```

## 快速查看 profile，不启动训练

```bash
DRY_RUN=1 bash shell/run_locany_ui_defect.sh 4090
DRY_RUN=1 bash shell/run_locany_ui_defect.sh h20
DRY_RUN=1 bash shell/run_locany_ui_defect.sh a800
```

这适合在提交任务前核对最终生效的路径、GPU 数、冻结策略和 token 预算。

## 样例数据格式

训练 recipe 位于：

```text
samples/ui_defect_locany_smoke_real/recipe/ui_defect_5class_train.json
```

每个 JSONL 样本与正式数据保持相同格式：

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Locate all the instances that match the following description: text overflow."
    },
    {
      "from": "gpt",
      "value": "<ref>text overflow</ref><box><281><363><922><525></box>"
    }
  ],
  "image": "ui_text_overflow_positive.png"
}
```

正样本坐标是 `[0, 1000]` 归一化坐标；负样本答案为 `<box>none</box>`。recipe 的 annotation 与 image root 都是相对 `Embodied` 根目录的路径，因此仓库移动到任何机器后都不需要改 JSON。

如需重建合成样例：

```bash
python scripts/generate_ui_defect_locany_smoke.py --force
python scripts/validate_ui_defect_locany_sample.py
```

## 如确实需要导出 10 条真实样例

公司内网截图可能含用户信息、业务信息或受限素材。外传前必须确认数据具有外网使用和提交权限。默认合成样例已经足以验证训练管线；没有授权时不要把真实样例放进外网仓库。

获得明确授权后，可在内网按五类各抽取一条正样本和一条负样本，并自动复制图片、改写为相对路径：

```bash
python scripts/export_ui_defect_locany_smoke.py \
  --source-data-dir data/ui_defect_locany_v3 \
  --output-dir samples/ui_defect_locany_smoke_real \
  --confirm-authorized-export
```

导出脚本不会覆盖已有目录，也不会自动执行 Git 操作。导出后仍需人工逐图检查脱敏情况，再决定是否移动或提交。

## 建议的 Git 工作流

只同步代码、配置和合成样例：

```bash
git add \
  README_UI_DEFECT_DEV.md \
  samples/ui_defect_locany_smoke_real \
  scripts/generate_ui_defect_locany_smoke.py \
  scripts/validate_ui_defect_locany_sample.py \
  scripts/export_ui_defect_locany_smoke.py \
  shell/run_locany_ui_defect.sh \
  shell/train_locany_ui_defect.sh \
  jobs/locany_ui5_v3_a800x8_merlin.yaml

git commit -m "add portable UI defect smoke training profiles"
git push
```

不要提交以下内容：模型权重、checkpoint、TensorBoard 日志、真实全量数据、内网缓存和带身份信息的截图。

## 两点需要保留的边界

第一，4090 profile 验证的是工程可运行性，不验证正式全参训练的显存、吞吐和收敛。模型结构改动如果只在 H20/Magi 专属分支内触发，仍需在内网补一次 1–2 step 的 H20 冒烟测试。

第二，`nvidia/LocateAnything-3B` 官方模型卡标注为非商业研究许可。若在公司环境中训练、部署或用于业务实验，应先确认使用场景符合模型许可。官方资料：

- https://github.com/NVlabs/Eagle/tree/main/Embodied
- https://github.com/NVlabs/Eagle/blob/main/Embodied/document/TRAINING.md
- https://github.com/NVlabs/Eagle/blob/main/Embodied/document/DATA_PREPARATION.md
- https://huggingface.co/nvidia/LocateAnything-3B

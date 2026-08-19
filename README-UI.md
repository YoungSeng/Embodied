# LocateAnything：五类 UI 缺陷微调

目标目录：

```text
/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied/
├── scripts/prepare_ui_defect_locany.py
├── shell/train_locany_ui_defect.sh
├── data/ui_defect_locany/
└── recipe/ui_defect_5class_train.json
```

## 1. 安装 Eagle

```bash
# A800
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/

git clone https://github.com/NVlabs/Eagle.git Eagle
cd ./code/Eagle/Embodied
conda create -n LocateAnything python=3.12 -y
conda activate LocateAnything
pip install -e .

# optional
git clone https://github.com/SandAI-org/MagiAttention.git
cd MagiAttention
git checkout v1.0.5
git submodule update --init --recursive
pip install -r requirements.txt

nproc
free -h
> CPU ≤ 16 核： MAX_JOBS=8
> CPU 16–32 核：MAX_JOBS=16
> CPU 32–64 核：MAX_JOBS=24 或 32
> CPU > 64 核： 一般仍先用 32

python -m pip install -U nvidia-ml-py
python -m pip install -U sortedcontainers
python -m pip install -U tensorboard
python -m pip install -U debugpy

MAGI_ATTENTION_PREBUILD_FFA_JOBS=2 \
MAX_JOBS=4 \
NVCC_THREADS=2 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python -m pip install --no-build-isolation -v .

MAX_JOBS=4 python -m pip install --no-build-isolation -v .
```

H20 使用 Magi Attention 时，按官方说明安装 MagiAttention v1.0.5。A800 走
SDPA，无需安装 MagiAttention，但上下文长度应限制在约 4096。

## 2. 放置脚本

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied
mkdir -p scripts shell recipe data
cp /path/to/prepare_ui_defect_locany.py scripts/
cp /path/to/train_locany_ui_defect.sh shell/
chmod +x scripts/prepare_ui_defect_locany.py shell/train_locany_ui_defect.sh
```

## 3. 先做 100 条/文件的冒烟转换

```bash
python scripts/prepare_ui_defect_locany.py \
  --project-root /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied \
  --source-dir /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data \
  --label-style bilingual \
  --val-ratio 0.02 \
  --max-samples-per-file 100 \
  --strict
```

检查：

```bash
python -m json.tool \
  data/ui_defect_locany/conversion_summary.json \
  | less

head -n 1 data/ui_defect_locany/ui_occlusion_train.jsonl \
  | python -m json.tool
```

## 4. 转换全部五个文件

```bash
python scripts/prepare_ui_defect_locany.py \
  --project-root /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied \
  --source-dir /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data \
  --label-style bilingual \
  --val-ratio 0.02 \
  --negative-keep-ratio 1.0 \
  --strict \
  2>&1 | tee data/ui_defect_locany/prepare.log
```

转换脚本会：

- 将 `messages/images/objects` 转成 LocateAnything 的 `conversations/image`；
- 将 `bbox_type=real` 的像素框转成 `<0>` 到 `<1000>`；
- 正样本写成 `<ref>类别</ref><box>...</box>`；
- 负样本写成 `<box>none</box>`；
- 按图片路径进行 train/val 哈希划分，避免同一截图的五类样本跨集合；
- 生成训练 recipe 和验证 recipe。

## 5. 8 张 H20 训练

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=nvidia/LocateAnything-3B \
RUN_NAME=locany-3b-ui5-h20-full-v1 \
REPORT_TO=tensorboard \
bash shell/train_locany_ui_defect.sh
```

使用本地模型：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MODEL_PATH=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/llm-models/LocateAnything-3B \
RUN_NAME=locany-3b-ui5-h20-full-v1 \
bash shell/train_locany_ui_defect.sh
```

## 6. 4 张 H20 训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=nvidia/LocateAnything-3B \
RUN_NAME=locany-3b-ui5-h20-4gpu-full-v1 \
bash shell/train_locany_ui_defect.sh
```

脚本会自动把梯度累积设为 2，使四卡和八卡的 rank-batch 数量接近。

## 7. 4 或 8 张 A800 训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=nvidia/LocateAnything-3B \
RUN_NAME=locany-3b-ui5-a800-full-v1 \
ATTN_IMPLEMENTATION=sdpa \
MAX_SEQ_LENGTH=4096 \
MAX_NUM_TOKENS_PER_SAMPLE=4096 \
MAX_NUM_TOKENS=4096 \
bash shell/train_locany_ui_defect.sh
```

A800 显存不足时依次降低：

```bash
PACKING_BUFFER_SIZE=8
DATALOADER_NUM_WORKERS=1
```

若仍然 OOM，优先改用 H20；不要在 A800 上把 SDPA 的序列长度提高到 8K/16K。

## 8. W&B

不要把 API Key 写进脚本。需要 W&B 时：

```bash
wandb login
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
REPORT_TO=wandb \
WANDB_PROJECT=locateanything-ui-defect \
RUN_NAME=locany-3b-ui5-h20-full-v1 \
bash shell/train_locany_ui_defect.sh
```

## 9. 推理标签

默认 `--label-style bilingual`，因此推理时建议使用同样的五个描述：

```python
[
    "元素重叠 (overlapping elements)",
    "元素被裁切 (cropped element)",
    "文字溢出容器 (text overflow)",
    "文字省略异常 (abnormal text ellipsis)",
    "内容未展示 (missing content)",
]
```

输出框数量就是问题数量；无需再让模型额外生成“n 个问题”。






如果遇到`ImportError: Batch inference requires the Hugging Face release files `batch_utils/` and `kernel_utils/` on PYTHONPATH. Download nvidia/LocateAnything-3B and run from that directory, or add the model directory to PYTHONPATH.`

```
cd ./code/Eagle/Embodied/

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/LocateAnything-3B",
    local_dir=".",
    allow_patterns=[
        "batch_utils/**",
        "kernel_utils/**",
    ],
)

print("Downloaded batch_utils/ and kernel_utils/ to current directory.")
PY
```




## Model

```
hf download nvidia/LocateAnything-3B --local-dir models/LocateAnything-3B
```


## Data 

```
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied

python scripts/prepare_ui_defect_locany.py \
  --project-root /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied \
  --source-dir /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data \
  --label-style bilingual \
  --prompt-language en \
  --val-ratio 0.1 \
  --negative-keep-ratio 1.0 \
  --bbox-format auto \
  --bbox-coord-mode xyxy \
  --strict \
  2>&1 | tee data/ui_defect_locany/prepare.log

# ==================== 1. 根据机器切换路径 ====================

# A800
ROOT_PATH="/mnt/bn/intelligent-service-yg/logging/sicheng_workspace"

# H20
# ROOT_PATH="/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace"

VERSION="v3"   # 修改为 v2 或 v3

PROJECT_ROOT="${ROOT_PATH}/code/Eagle/Embodied"
SOURCE_DIR="${ROOT_PATH}/data"
V_DIR="${PROJECT_ROOT}/data/ui_defect_locany_${VERSION}"

mkdir -p "${V_DIR}/recipe"

python scripts/prepare_ui_defect_locany.py \
    --project-root "${PROJECT_ROOT}" \
    --source-dir "${SOURCE_DIR}" \
    --label-style en \
    --prompt-language en \
    --val-ratio 0.0 \
    --negative-keep-ratio 1.0 \
    --bbox-format auto \
    --bbox-coord-mode xyxy \
    --strict \
    --output-dir "${V_DIR}" \
    --recipe-dir "${V_DIR}/recipe" \
    2>&1 | tee "${V_DIR}/prepare_${VERSION}.log"

```

## Train

```

export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1

# A800
# ROOT_PATH="/mnt/bn/intelligent-service-yg/logging/sicheng_workspace"
# MACHINE_NAME="a800"
# CUDA_DEVICES="0,1,2,3,4,5,6,7"

# H20
ROOT_PATH="/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace"
MACHINE_NAME="h20"
CUDA_DEVICES="0,1,2,3"

if [[ "${MACHINE_NAME}" == "h20" ]]; then
  ATTN_IMPLEMENTATION="magi"
  MAX_SEQ_LENGTH=7268
  MAX_NUM_TOKENS_PER_SAMPLE=7268
  MAX_NUM_TOKENS=25600
  PACKING_BUFFER_SIZE=32
  DATALOADER_NUM_WORKERS=4
elif [[ "${MACHINE_NAME}" == "a800" ]]; then
  ATTN_IMPLEMENTATION="sdpa"
  MAX_SEQ_LENGTH=7268
  MAX_NUM_TOKENS_PER_SAMPLE=7268
  MAX_NUM_TOKENS=25600
  PACKING_BUFFER_SIZE=32
  DATALOADER_NUM_WORKERS=4
else
  echo "[ERROR] Unsupported MACHINE_NAME=${MACHINE_NAME}" >&2
  exit 1
fi

DATA_VERSION="v3"
VERSION="v3_h20x4"

PROJECT_ROOT="${ROOT_PATH}/code/Eagle/Embodied"
OUTPUT_BASE="${ROOT_PATH}/gui_models"
HF_HOME="${ROOT_PATH}/cache/huggingface"
DATA_DIR="${PROJECT_ROOT}/data/ui_defect_locany_${DATA_VERSION}"
META_PATH="${DATA_DIR}/recipe/ui_defect_5class_train.json"
RUN_NAME="locany-3b-ui5-${MACHINE_NAME}-full-${VERSION}-en"

cd "${PROJECT_ROOT}"
MODEL_DIR="${ROOT_PATH}/models/LocateAnything-3B"

LAUNCHER=pytorch \
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
OUTPUT_BASE="${OUTPUT_BASE}" \
HF_HOME="${HF_HOME}" \
MODEL_PATH="${MODEL_DIR}" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_DISABLE_XET=1 \
META_PATH="${META_PATH}" \
RUN_NAME="${RUN_NAME}" \
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}" \
MAX_STEPS=25000 \
WARMUP_STEPS=500 \
LEARNING_RATE=2e-5 \
LOGGING_STEPS=5 \
SAVE_STEPS=2000 \
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH}" \
MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE}" \
MAX_NUM_TOKENS="${MAX_NUM_TOKENS}" \
PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE}" \
SAVE_TOTAL_LIMIT=1000 \
SAMPLE_LOG_INTERVAL=5 \
REPORT_TO=tensorboard \
SAVE_EVERY_N_HOURS=0 \
bash shell/train_locany_ui_defect.sh

grep "\[SampleStats\]" \
  /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_models/locany-3b-ui5-h20-full-v1/train-20260728-202359.log
```

Avg samples/step ≈ 4.10
表示 rank 0 每个 optimizer step 平均消耗约 4.1 条原始样本。官方代码在每个 training_step 中统计 sub_sample_lengths，而日志只在 rank 0 输出；因为你有梯度累积 2，所以同一个 step 会打印两次
四张卡每次参数更新大约处理：4.10 × 4 GPU = 16.4 条原始样本
完整使用约 78,940 条训练样本需要：78,940 ÷ 16.4 ≈ 4,813 steps

## inference

```
# A800
# BASE_MODEL="/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/hf_home/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0"
# OUTPUT_DIR="/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_models/locany-3b-ui5-a800-full-v3-en"

# H20
BASE_MODEL="/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/hf_home/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca5e996f5a7a485845b4f57a233936bba0"
OUTPUT_DIR="/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_models/locany-3b-ui5-h20-full-v3_h20x4-en"

for ckpt in "${OUTPUT_DIR}"/checkpoint-*; do
    [ -d "${ckpt}" ] || continue
    echo "Patching ${ckpt}"
    cp -L "${BASE_MODEL}"/*.py "${ckpt}/"
done
```

```
# A800
# WORKSPACE=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace

# H20
WORKSPACE=/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace
PROJECT_ROOT=${WORKSPACE}/code/Eagle/Embodied

cd "${PROJECT_ROOT}"

# A800
python scripts/inference_ui_defect_locany.py \
  --checkpoint "${WORKSPACE}/gui_models/locany-3b-ui5-a800-full-v3-en/checkpoint-18000" \
  --input-dir "${WORKSPACE}/data" \
  --output-dir "${WORKSPACE}/gui_models/locany-3b-ui5-a800-full-v3-en/inference-checkpoint-18000-full" \
  --cuda-visible-devices 0 \
  --device cuda:0 \
  --attn-implementation sdpa \
  --generation-mode hybrid \
  --skip-figma


  --tag-filename \
  --save-raw-answer \
  --save-visualization

# H20
ENV_DIR="${WORKSPACE}/conda_envs/LocateAnything"

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONNOUSERSITE=1
hash -r

which python
python -V

python scripts/inference_ui_defect_locany.py \
  --checkpoint "${WORKSPACE}/gui_models/locany-3b-ui5-h20-full-v3_h20x4-en/checkpoint-2000" \
  --input-dir "${WORKSPACE}/data" \
  --output-dir "${WORKSPACE}/gui_models/locany-3b-ui5-h20-full-v3_h20x4-en/inference-checkpoint-2000-full" \
  --cuda-visible-devices 3 \
  --device cuda:0 \
  --attn-implementation magi \
  --generation-mode hybrid \
  --skip-figma

```

处理：tag-filename

```
find /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_models/locany-3b-ui5-a800-full-v3-en/inference-checkpoint-4000-full/ \
  -type f \( -name '*_ok.json' -o -name '*_defect.json' \) -print0 | \
while IFS= read -r -d '' f; do
  new="${f%_ok.json}.json"
  [[ "$f" == *_defect.json ]] && new="${f%_defect.json}.json"
  mv -f -- "$f" "$new"
done
```

同步最新的内容：
```
rsync -av --progress \
  /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied-master/ \
  /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied/
```
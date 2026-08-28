# LocateAnything UI CPT v2

本目录只说明 CPT。CPT 运行必须设置 `LOCANY_CPT_MODE=1`；新增统计、采样校验和
UI5 检查旁路均受此开关保护，不改变 SFT 默认行为。

UI CPT v2 的目标是把训练改造成可观测、可诊断、可比较的系统：按图片组切分
held-out、精确核对样本与监督 token、按任务计算 train/eval CE、用任务专属指标评估，
并让 JSON/JSONL 成为唯一数据源。Excel 只是可选离线投影。

## 0. 当前正式启动流程：A800 完整验收，H20 直接 formal

当前执行策略固定为：

```text
YG smoke recipe
→ YG A800×4 checkpoint-10→20 smoke
→ YG A800 held-out eval
→ YG A800 validator passed
→ YG 全量 recipe 检查
→ HL 全量 recipe
→ H20×4 formal
→ 每训练约 6 小时自动 checkpoint → 同 job 单卡 held-out generation/CE → 自动 resume
```

H20 排队通常需要一到两天，因此不再把 H20 smoke/eval 作为 formal 启动前门禁。H20
四卡使用固定的 SDPA + 7268/7268/7268 + packing buffer 16 + 梯度累积 2。仓库命令中的
`a100` 是历史 profile 标识；对应 Merlin YAML 实际申请 `A800_SXM_40GB`，不是 A100。

下面各命令按顺序逐条执行。不要把 YG 生成的 recipe 复制到 HL；两边 JSONL 的绝对图片
路径不同，必须分别生成。

### 0.1 YG：生成 A800 smoke recipe

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied-CPT
```

当前已经有一次失败准备留下的部分文件，所以本次重跑使用 `OVERWRITE=1`：

```bash
OVERWRITE=1 \
SOURCE_ROOT=/mnt/bn/intelligent-service-yg/dataset/gui/gui_base/sample/raw_data_v4.1_hl_norm1k/raw_data_v4.1_hl \
DATA_DIR=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/locany_cpt_v4_split_v2_smoke \
bash shell/prepare_locany_cpt_v2.sh a100 smoke
```

成功标志是最后出现 `CPT_V2_DATA_READY=...`，且每任务打印 written、known_dropped、
rejected。已确认的 ref/box 换行异常和退化框计入 `known_dropped`；其他异常仍计入
`rejected`。训练启动后禁止再次覆盖这个目录。

### 0.2 YG：提交四卡 A800 smoke

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied-CPT
```

用 `--cluster` 选择 A800 调度资源组。普通 YG 资源组使用：

```bash
/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything/bin/python \
scripts/submit_locany_cpt.py --mode smoke --cluster yg
```

如果希望改到 `ies_aiai_experience/AIAI_locate` 排队，只执行下面这条，不要与上一条同时提交：

```bash
/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything/bin/python \
scripts/submit_locany_cpt.py --mode smoke --cluster aiai_locate
```

`yg` 对应 group 1602 和默认 queue；`aiai_locate` 对应 group 2146 和
`compute-3302-yg-cloudnative-ai-aiai.locate-guarantee`。两者仍属于 YG、挂载同一个
`/mnt/bn/intelligent-service-yg`，所以 recipe、图片路径和 `RUN_NAME` 都不需要修改。
提交器会把最终 YAML 写到 `jobs/rendered/`，打印 group/queue 后再调用
`mlx job submitv2`。只想检查 YAML 而不提交时追加 `--render-only`。

该 job 实际申请 `A800_SXM_40GB × 4`，先保存 checkpoint-10，再从 checkpoint-10
自动 resume 到 checkpoint-20。必须等 job 正常完成后再执行下一步。

如果旧 job 已在 checkpoint-10 完整写盘后因 completion marker/eval queue 的
`Errno 38: Function not implemented` 退出，直接用相同命令重新提交，不要删除
checkpoint-10。launcher 会先做 resume 完整性验证；验证通过时跳过前十步，训练入口自动
补齐 marker/queue，并从 checkpoint-10 继续到 checkpoint-20。

### 0.3 YG：在 A800 上跑 held-out eval

在可使用 A800 的节点执行。评测只使用 GPU 0，因此显式限制可见卡数为 1：

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied-CPT
```

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_DIR=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_models/locany-3b-ui-cpt-v4-v2-a100x4-smoke \
DATA_DIR=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/locany_cpt_v4_split_v2_smoke \
EVAL_MAX_PENDING=2 \
EVAL_SAMPLES_PER_TASK=10 \
bash shell/run_locany_cpt_eval_merlin.sh a100
```

这个命令依次消费 checkpoint-10 和 checkpoint-20，Base 结果只计算一次并缓存。成功时
十任务 Base/checkpoint teacher-forced CE 均非空、inference error 为 0，并生成真实
`cpt_eval_metrics.jsonl` 和三 sheet Excel。

这里的 `EVAL_SAMPLES_PER_TASK=10` 是每个模型 100 条，不是整个命令只跑 100 条：首次
执行需要完成 Base 100 条、checkpoint-10 100 条、checkpoint-20 100 条，共约 300 次
generation + teacher-forced forward。Base 的 100 条全部完成后才原子写入 cache；在此之前
按 Ctrl-C 会丢失本轮尚未落盘的 Base 结果，下次需要从 Base 第 1 条重跑。单卡 `slow`
generation 且每条最多生成 1024 token，耗时可能从几十分钟到数小时，不能用模型加载后的
短暂无输出来判断卡死。

新 evaluator 默认每条打印 `[EVAL] ... START/DONE`；单条超过 60 秒时还会打印
`[EVAL HEARTBEAT]`，其中包含当前 Base/checkpoint、样本序号、task、phase 和耗时。可用
`--progress-every N` 降低 START/DONE 频率，用 `--progress-heartbeat-seconds N` 修改
heartbeat 间隔（设为 0 才会关闭）。运行期间可在另一终端检查 GPU 和进程：

```bash
watch -n 5 'nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader'
```

```bash
PID=$(pgrep -n -f 'scripts/eval_locany_cpt_learning.py')
ps -p "$PID" -o pid,etime,%cpu,%mem,stat,cmd
```

GPU 利用率/功耗周期性变化，或进程仍有 CPU 活动时继续等待。不要并行启动第二个 consumer，
否则它会等待同一个 Base cache 锁。只有 heartbeat 长时间停在同一 phase，且 GPU、CPU 都
持续无活动时，才按异常排查。

如果 eval 因基础设施或依赖问题退出，对应 queue row 会变为 `failed`。修复代码后在同一
命令中增加 `EVAL_RETRY_FAILED=1`；runner 会显式重试 failed row，Base cache 和指标写入
均使用与 eval queue 相同的 ByteNAS 兼容锁。

例如 checkpoint-10 已标记为 failed 时，先确认当前目录就是同步了最新代码的 clone
（若实际使用 `Eagle_LocateUI5_v4/Embodied-CPT`，就进入该目录），再完整执行：

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-CPT
```

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_DIR=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_models/locany-3b-ui-cpt-v4-v2-a100x4-smoke \
DATA_DIR=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/locany_cpt_v4_split_v2_smoke \
EVAL_MAX_PENDING=2 \
EVAL_SAMPLES_PER_TASK=10 \
EVAL_RETRY_FAILED=1 \
bash shell/run_locany_cpt_eval_merlin.sh a100
```

新日志应出现
`generation implementation: eaglevl.utils.locany.modeling_locateanything (repository)`；
Base 和当前 CPT evaluator 还应打印 `generation UI relation : False`；
评测协议 v4 会自动使用新的 Base cache key，不会复用旧版失败缓存。若仍失败，日志会明确
打印 `phase=generation|teacher_forced` 以及模型内部完整 traceback，不能再只保留顶层
`NoneType` 文本。

teacher-forced forward 会关闭 KV cache，并仅将 Qwen decoder 容器临时切到训练时的
packed/causal mask 构造分支；其 attention、MLP 和 dropout 子层仍保持 eval，forward
结束后 decoder 状态也会恢复。这个兼容处理用于解决旧 Base remote code 在
`inputs_embeds` 路径错误下标访问 `input_ids=None` 的问题，不会改变训练代码。

ByteNAS 的目录锁现在写入 `owner.json`（host/PID/token），并持续刷新 heartbeat。同主机
owner PID 已退出时立即回收；跨主机 owner 仅在 heartbeat 超过 stale 阈值后回收。旧版本
遗留的空 `.lock.mkdir` 没有 owner，默认不会贸然抢占；确认没有 evaluator 进程后可用
精确 `rmdir` 清理，禁止对 run 目录执行递归删除。

### 0.4 YG：执行 A800 最终 validator

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied-CPT
```

```bash
/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything/bin/python \
scripts/validate_locany_cpt_smoke.py \
--run a800=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_models/locany-3b-ui-cpt-v4-v2-a100x4-smoke \
--require-eval \
--eval-samples-per-task 10 \
--output /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_models/locany-3b-ui-cpt-v4-v2-a100x4-smoke/diagnostics/cpt_smoke_validation.json
```

只有命令零退出且输出包含下面内容，才允许进入 HL formal：

```json
"status": "passed"
```

### 0.5 YG：额外生成并全量检查 formal recipe

这一步不启动 A800 formal，只利用 YG 数据与图片路径对全量数据做转换、图片存在性检查、
group split 和零泄漏验证：

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle/Embodied-CPT
```

```bash
SOURCE_ROOT=/mnt/bn/intelligent-service-yg/dataset/gui/gui_base/sample/raw_data_v4.1_hl_norm1k/raw_data_v4.1_hl \
DATA_DIR=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/locany_cpt_v4_split_v2 \
bash shell/prepare_locany_cpt_v2.sh a100 formal
```

如果此前的 full recipe 准备也曾中途失败，确认没有任务使用该目录后，再在上述命令前添加
`OVERWRITE=1`。

### 0.6 HL：生成 H20 formal recipe

先确保相同版本代码已经同步到 HL，然后执行：

```bash
cd /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Eagle/Embodied-CPT
```

```bash
SOURCE_ROOT=/mnt/bn/intelligent-service-arnold-hl/dataset/gui/gui_base/sample/raw_data_v4.1_hl_norm1k/raw_data_v4.1_hl \
DATA_DIR=/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/data/locany_cpt_v4_split_v2 \
bash shell/prepare_locany_cpt_v2.sh h20 formal
```

必须看到 `CPT_V2_DATA_READY=...` 后再提交训练。

全量准备在最后一个任务（通常日志停在 `vqa read=...`）之后还没有结束。随后会执行：归一化
文件 flush/原子发布、第一遍逐图片内容 SHA-256 与跨任务连通分组、第二遍聚合 group/标签
分层、第三遍写 train/val/manifest，最后生成 val_fast。第一次读取 NAS 图片内容可能明显比
JSONL 转换更久，这是正常阶段，不要看到最后一条 VQA 日志就 Ctrl-C。新日志会持续打印：

```text
[prepare] phase=group_level_split state=START mode=sha256
[split] phase=hash_images_and_connect_groups state=PROGRESS rows=...
[split] phase=aggregate_groups_and_strata state=PROGRESS rows=...
[split] phase=write_train_val_and_manifest state=PROGRESS rows=...
```

第一遍被 Ctrl-C 时会尽量保存 `diagnostics/image_hash_cache.json`；确认没有另一个 prepare
进程后使用同一命令加 `OVERWRITE=1` 重跑，可复用已完成的图片哈希。只有最终出现
`CPT_V2_DATA_READY=...` 才算 recipe 完成。

### 0.7 HL：直接提交 H20×4 formal（含每 6 小时集成评测）

```bash
cd /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Eagle/Embodied-CPT
```

```bash
mlx job submitv2 --path locany_cpt_v4_h20x4_formal_merlin.yaml
```

这个主流程不再先提交 H20 smoke。formal 使用新的
`RUN_NAME=locany-3b-ui-cpt-v4-v2-h20x4-formal`，不会续训 checkpoint-1549/1860。

该 YAML 固定 `CPT_INTEGRATED_EVAL=1`、`SAVE_EVERY_N_HOURS=6`。一个 segment 约训练
6 小时后，四卡 torchrun 先保存完整可续训 checkpoint 并正常退出以释放显存；同一个
Merlin job 随后只暴露 GPU 0，真实运行固定 held-out val_fast 的 generation 和
teacher-forced CE。评测成功后自动从刚才的 checkpoint 恢复四卡训练。评测时间不计入下一个
6 小时训练 interval，因此相邻评测结果的实际墙钟间隔是“约 6 小时 + 上一次评测耗时”。

评测失败时 formal job 非零退出，不会绕过测试继续训练；checkpoint 仍可完整 resume。修复
代码后重新提交同一 YAML，会先重试 queue 中 failed/pending 的评测，再开始下一段训练。

### 0.8 HL：人工补跑 held-out eval（仅用于故障恢复）

主流程不需要提交独立 eval job。只有集成评测曾失败、或需要人工重算历史 checkpoint 时才提交：

```bash
cd /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Eagle/Embodied-CPT
```

```bash
mlx job submitv2 --path locany_cpt_v4_h20x1_eval_merlin.yaml
```

该 eval job 一次最多消费当前积压的 20 个 checkpoint。Base 结果按同一 manifest/protocol
缓存，不会为每个 checkpoint 重复推理。

## 1. 数据准备与无泄漏切分

默认按图片内容 SHA-256 生成 `group_id`，固定 seed `20260826`，以 group 为单位近似
98%/2% 切分。同一图片派生的不同任务和 annotation row 只能进入同一 split；若多图记录
与其他记录共享任意一张图片，会按共享图片的连通分量合并为同一 group，避免局部重叠泄漏。
只有在全量内容哈希确实不可承受时才使用 `--group-id-mode path`；该模式会额外输出
`diagnostics/path_duplicate_suspects.json`，列出同 basename、同尺寸但路径不同的候选，
必须抽样复核，不能将它当成与内容哈希同等强度的零泄漏证据。

```bash
python scripts/prepare_locany_cpt.py \
  --source-root /path/to/raw_data_v4.1_hl \
  --output-dir /path/to/locany_cpt_v4_split_v2 \
  --recipe-name locany_cpt_train.json \
  --split-seed 20260826 \
  --val-fraction 0.02 \
  --val-fast-per-task 200 \
  --group-id-mode sha256 \
  --overwrite
```

集群上不要复用上一轮的 `locany_cpt_v4`。仓库提供了可直接生成并完整校验
train/val/val_fast、manifest 的命令；smoke 默认每任务先取 2000 条、val_fast 至少 10 条：

```bash
bash shell/prepare_locany_cpt_v2.sh a100 smoke
bash shell/prepare_locany_cpt_v2.sh a100 formal
bash shell/prepare_locany_cpt_v2.sh h20 formal
```

已有同名 v2 目录时命令会拒绝覆盖；只有明确要重建时才设置 `OVERWRITE=1`。A800
文件系统使用 `a100` 参数；若原始数据不在默认位置，显式设置 `SOURCE_ROOT`。

输出包括：

- `train/<task>.jsonl`、`val/<task>.jsonl`；
- `recipe/locany_cpt_train.json`、`locany_cpt_val.json`、
  `locany_cpt_val_fast.json`；
- `diagnostics/split_manifest.jsonl`、`split_summary.json`、图片哈希缓存；
- 被格式转换拒绝的行及真实原因。

数据中两类已确认无法形成有效监督的 annotation 会直接排除：非规范的
`<ref>...</ref><box>...</box>` 配对（包括换行导致的 ref/box 脱离），以及归一化后宽或高
为 0 的退化框。它们仍逐条写入 `rejected.jsonl`，标记
`disposition=known_data_drop` 和具体 `category`，并汇总为 manifest 的
`total_known_dropped/known_drop_rate`；这两类不计入 `--max-error-rate`。其他
NormalizeError 继续计入 `total_rejected/rejected_rate`，未知运行时异常仍直接失败。

若已有规范化的未切分 recipe，可单独切分：

```bash
python scripts/split_locany_cpt.py \
  --recipe /path/to/locany_cpt_all.json \
  --output-dir /path/to/locany_cpt_v4 \
  --seed 20260826 \
  --val-fraction 0.02 \
  --val-fast-per-task 200
```

全量验证必须读取 manifest；发现 group、record、图片内容哈希或规范化路径跨 split 时
非零退出：

```bash
python scripts/validate_locany_cpt.py \
  --recipe /path/to/locany_cpt_v4/recipe/locany_cpt_train.json \
  --records-per-dataset 0 \
  --split-manifest /path/to/locany_cpt_v4/diagnostics/split_manifest.jsonl \
  --require-split train \
  --require-equal-weights

python scripts/validate_locany_cpt.py \
  --recipe /path/to/locany_cpt_v4/recipe/locany_cpt_val.json \
  --records-per-dataset 0 \
  --split-manifest /path/to/locany_cpt_v4/diagnostics/split_manifest.jsonl \
  --require-split heldout
```

`val_fast` 不是取文件前 N 条，而是按 `seed + group_id + record_id` 的 hash 固定选择；
VQA 和 UI defect 保持标签分层。

## 2. 静态长度与采样模拟

在训练前用真实 processor 统计 pre/post-MTP 长度、main/MTP token 及超限原因，并把
每任务平均监督 token 写回 recipe，供 token-aware 策略使用：

```bash
python scripts/analyze_locany_cpt_lengths.py \
  --recipe /path/to/locany_cpt_v4/recipe/locany_cpt_train.json \
  --processor /path/to/LocateAnything-3B \
  --output /path/to/locany_cpt_v4/diagnostics/cpt_data_stats.json \
  --block-size 6 \
  --max-num-tokens-per-sample 8192
```

静态分析同时写 `diagnostics/oversize_samples.jsonl`，逐条保存 task、record/group、
source/line、pre/post-MTP 长度和超限原因。训练 JSONL 的
`window_oversize_record_hashes`/`window_oversize_group_hashes` 记录该统计窗口首次出现的
真实 runtime skip，可与 manifest 的稳定 ID 对照且 resume 后不会重复写。

离线比较四种固定采样；这不会启动训练：

```bash
python scripts/simulate_locany_cpt_sampling.py \
  --recipe /path/to/locany_cpt_v4/recipe/locany_cpt_train.json \
  --data-stats /path/to/locany_cpt_v4/diagnostics/cpt_data_stats.json \
  --output /path/to/locany_cpt_v4/diagnostics/cpt_sampling_simulation.json \
  --optimizer-steps 20000 \
  --samples-per-step 23.57
```

采样统一使用 `p_i ∝ N_i^alpha × mu_i^(-beta)`：

| `CPT_SAMPLING_MODE` | alpha | beta | 含义 |
| --- | ---: | ---: | --- |
| `sample_equal` | 0 | 0 | 任务样本等权，formal 默认 |
| `sqrt_size` | 0.5 | 0 | 增加大任务覆盖 |
| `token_balanced` | 0 | 1 | 近似任务监督 token 等权 |
| `hybrid` | 0.5 | 0.5 | 覆盖与 token 平衡折中 |

可覆盖 `CPT_SIZE_ALPHA`、`CPT_TOKEN_BETA`、`CPT_MIN_TASK_PROB`、
`CPT_MAX_TASK_PROB`。概率在训练前固定并写入 run config；resume 时必须一致，训练中
不会根据 rolling statistics 动态改权重。

## 3. 训练和 smoke

第一阶段 formal 保持 `CPT_SAMPLING_MODE=sample_equal`：

```bash
CPT_SAMPLING_MODE=sample_equal \
RUN_NAME=locany-3b-ui-cpt-v4-v2-h20x4-formal \
CPT_INTEGRATED_EVAL=1 \
SAVE_EVERY_N_HOURS=6 \
bash shell/run_locany_cpt_merlin.sh h20 formal
```

Merlin 入口：

```bash
/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything/bin/python \
  scripts/submit_locany_cpt.py --mode smoke --cluster yg
/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything/bin/python \
  scripts/submit_locany_cpt.py --mode formal --cluster yg
mlx job submitv2 --path locany_cpt_v4_h20x4_formal_merlin.yaml
```

上述两条 A800 命令都可将 `--cluster yg` 改为 `--cluster aiai_locate`。该参数只控制
调度资源组，不切换代码、ByteNAS、数据或 recipe。第一阶段流程仍只需要 smoke；不要因为
这里给出了 A800 formal 命令就同时启动另一场正式实验。

当前 formal 启动前只要求 A800 smoke：它先在 step 10 强制保存并退出 segment，再从同一
checkpoint 自动 resume 到 step 20，因此一个 job 同时覆盖多卡训练与断点续训。H20×4
formal 默认每 rank packed-token 上限 7268、单样本与序列上限也为 7268、SDPA、
packing buffer 16、梯度累积 2；A800×4 默认 SDPA + 7268/7268/12800、梯度累积 2。
`shell/run_locany_cpt.sh`
会先验证 train split，再将 split/length stats 复制到 run 的 `diagnostics/`。

这里三个数依次是 `MAX_SEQ_LENGTH`、`MAX_NUM_TOKENS_PER_SAMPLE`、
`MAX_NUM_TOKENS`。因此 A800 的单样本上限仍是 7268，12800 是每 rank 一个 packed batch
可容纳的总 token；当前 H20×4 三项均为 7268，不使用旧配置中的 25600。集成 evaluator
一次只处理一个样本，不受训练 packing 的 12800 控制；高分辨率 UI 图像必须让 MoonViT
使用 `flash_attention_2`，文本侧仍使用 SDPA。

H20 的 Magi 8192/7280 已实测 OOM，Magi 6400 已出现 gather index 越界，因此 v2 H20
smoke/formal 均不再把 Magi 或 25600 作为默认值。所有 v2 YAML 使用全新的
`DATA_DIR=.../locany_cpt_v4_split_v2[_smoke]` 与带 `v2` 的 `RUN_NAME`；新实验从原始
LocateAnything-3B checkpoint 开始，checkpoint-1549/1860 仅用于对照评测，禁止自动续训。

训练运行时只按区间写聚合统计，不逐样本打印：

- `CPT_METRICS_INTERVAL=100`：JSONL 和常规 logger/W&B/TensorBoard 标量；
- `CPT_TABLE_INTERVAL=500`：精确全局 unique coverage 和紧凑 per-task 表；
- stream 底层日志仍为每 10,000 packed batch。

## 4. 指标口径

样本计数满足：

```text
attempted_samples = accepted_samples + oversize_skipped_samples
```

`attempted` 是 iterator 取出的样本，`accepted` 是通过单样本 post-MTP 上限的样本，
`trained` 是实际进入 forward 的子样本，`packed_batches` 是产生的物理 packed batch。
图片读取、processor 或 token 对齐异常不会伪装成 oversize skip，而会记录 task/record/source
后抛出原始异常。

监督 token 满足：

```text
total_supervised_tokens = main_supervised_tokens + mtp_supervised_tokens
```

packed sequence 内每个 loss token 都保留 task id 和 supervision kind。统计 CE 使用同一次
forward 的 causal-shift unreduced CE，detach/no-grad 聚合，不做第二次 forward，也不改变
反向传播 loss：

- `train_main_token_ce`：main LM token CE；
- `train_mtp_token_ce`：MTP token CE；
- `train_total_token_ce`：两者按 token count 合并；
- held-out teacher-forced 使用 `eval_main_token_ce`；train–val gap 始终比较 main CE。

全局 CE 是 `sum(loss_sum) / sum(token_count)`，不是 rank mean 的均值。Unique record/group
通过可合并 64-bit hash set 全局 union，不能将各 rank unique 数直接相加。

覆盖率定义：

```text
effective_epoch = trained_samples / dataset_rows
repeat_factor   = trained_samples / max(unique_record_count, 1)
```

全局 oversize rate >0.5% 或任一任务 >2% 只报警；token share/sample share >2 标记为
`token_dominant`。不会自动扩大 token limit 或截断 ground truth。

## 5. Resume

Trainer checkpoint 保存 optimizer、scheduler、random state，以及每 rank 的
`dataloader_state_rank<N>.pt`。CPT resume 会校验：

- sampling config hash、各任务概率与 rows；
- world size、worker 数、base seed、单样本/packed token 上限、buffer size；
- metrics JSONL 不得领先所选 checkpoint；
- dataloader state 必须存在且版本兼容。

校验不通过时故意失败，避免静默从头取样造成重复计数。换 world size/worker 数或 sampling
策略应使用新的 `OUTPUT_DIR`。TorchElastic 入口带 `@record`，rank 失败保留原始异常、文件、
行号并以非零状态退出。

每个通过完整性校验的 checkpoint 还会由 rank 0 去重追加
`diagnostics/cpt_eval_queue.jsonl`，标记待同 job 集成阶段执行的 held-out val_fast 评测。完成标记
会先原子落盘、随后才发布 queue row，避免评测器抢到尚未完成的 checkpoint。queue row
具有 pending/running/completed/failed 状态；失败项只能显式设置 `EVAL_RETRY_FAILED=1` 重试。

## 6. Held-out 评测

训练池只能显式标记为 `train_pool/domain_absorption`；best checkpoint 只看 held-out。
H20×4 formal 默认每约 6 小时分段，在同一个 Merlin job 内释放训练进程后用 GPU 0 跑
generation + teacher-forced CE；评测完成才 resume。下面的直接命令只用于离线调试或重算：

```bash
python scripts/eval_locany_cpt_learning.py \
  --checkpoint "$CKPT" \
  --base-model "$BASE" \
  --processor-path "$BASE" \
  --recipe /path/to/locany_cpt_v4/recipe/locany_cpt_val_fast.json \
  --manifest /path/to/locany_cpt_v4/diagnostics/split_manifest.jsonl \
  --eval-split heldout \
  --subset-strategy hash \
  --samples-per-task 200 \
  --base-cache-dir "$RUN_DIR/eval/base_cache" \
  --train-metrics-jsonl "$RUN_DIR/diagnostics/cpt_train_metrics.jsonl" \
  --metrics-jsonl "$RUN_DIR/diagnostics/cpt_eval_metrics.jsonl" \
  --output-dir "$RUN_DIR/eval/$(basename "$CKPT")" \
  --device cuda:0 \
  --attn-implementation sdpa \
  --vision-attn-implementation sdpa \
  --teacher-forced
```

queue consumer 同时供集成评测和人工恢复使用。formal 启动前由 A800 smoke 完成
checkpoint-10→20 后的 held-out 门禁；在 A800 节点上只暴露 GPU 0：

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_DIR=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_models/locany-3b-ui-cpt-v4-v2-a100x4-smoke \
DATA_DIR=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/data/locany_cpt_v4_split_v2_smoke \
EVAL_MAX_PENDING=2 \
EVAL_SAMPLES_PER_TASK=10 \
bash shell/run_locany_cpt_eval_merlin.sh a100
```

formal 主流程不需要再提交单卡 job。Base 结果按 manifest/protocol 缓存，同一验证集不会
为每个 checkpoint 重复跑 Base。下面命令仅作为集成评测失败后的人工恢复入口：

```bash
mlx job submitv2 --path locany_cpt_v4_h20x1_eval_merlin.yaml
```

也可在已经分配的单卡机器上消费指定 run：

```bash
RUN_DIR=/path/to/run DATA_DIR=/path/to/locany_cpt_v4_split_v2 \
  bash shell/run_locany_cpt_eval_merlin.sh h20
```

每个 checkpoint 输出 `predictions.jsonl`、`summary.json`、
`errors_by_task.jsonl`、`qualitative_samples.md`，以及 50 条 referring_kg 人工语义复核模板。
原始 prediction 与解析结果足以离线升级指标：

```bash
python scripts/recompute_locany_cpt_metrics.py \
  --predictions "$RUN_DIR/eval/checkpoint-STEP/predictions.jsonl" \
  --output-summary "$RUN_DIR/eval/checkpoint-STEP/summary.recomputed.json"
```

primary metrics 分别为 caption ROUGE-L、agent action type+point、agent grounding point
Hit@50、class-aware defect Macro-F1@0.5、all-elements one-to-one F1@0.5、single
grounding Recall@0.5、OCR one-to-one label-aware F1@0.5、referring/referring_kg
ROUGE-L、VQA 正确/错误 accuracy；Char-F1 均作为附属指标。Point 支持两种语法且坐标必须位于 `[0,1000]`；box
采用一对一最大总 IoU 匹配，UI defect 只允许同类匹配。综合分数仅为十任务 primary 的
等权 `heldout_task_macro_primary`。

UI defect 的 hash 子集会在数据可用时强制覆盖文字溢出、文本省略、元素遮挡/重叠、元素
裁切、内容缺失五类。每次 Base/checkpoint 真实推理结束都会在日志打印五类的 image-level
P/R/F1 与 class-aware bbox P/R/F1@0.5，以及总的 image macro/micro F1 和 bbox
macro/micro F1@0.5。相同内容同时保存在 `summary.json`、`cpt_eval_metrics.jsonl` 和 Excel，
不是仅打印后丢失；bbox macro F1@0.5 仍是 ui_defect 的 primary metric，image F1 是额外
诊断指标。

best checkpoint 依次按 held-out task-macro primary 最大、task-macro main CE 最小排序；
关键弱任务相对历史 best 下降超过 3 个百分点时不自动标 best。少于完整十任务的 held-out
结果、train-pool 指标均不能成为 best。

分析 train–val 曲线：

```bash
python scripts/analyze_locany_cpt_curves.py \
  --train-metrics "$RUN_DIR/diagnostics/cpt_train_metrics.jsonl" \
  --eval-metrics "$RUN_DIR/diagnostics/cpt_eval_metrics.jsonl" \
  --output "$RUN_DIR/diagnostics/cpt_overfitting_analysis.json"
```

至少三个 milestone 才判定趋势；两点不足以声称“没有过拟合”。

## 7. JSON/JSONL 与 Excel

Source of truth 只有：

- `diagnostics/cpt_run_config.json`；
- `diagnostics/cpt_data_stats.json`；
- `diagnostics/cpt_train_metrics.jsonl`；
- `diagnostics/cpt_eval_metrics.jsonl`。

rank 0 原子写/追加这些文件。Excel 可随时离线重建：

```bash
python scripts/build_locany_cpt_excel.py \
  --diagnostics-dir "$RUN_DIR/diagnostics" \
  --output "$RUN_DIR/diagnostics/cpt_training_evaluation.xlsx"
```

工作簿严格只有三个 sheet，冻结表头并启用筛选：

- `TrainMetrics`：训练集的 step × task sample/token/skip/coverage/CE 记录；
- `EvalMetrics`：held-out 测试集的 checkpoint × task CE、primary、Base delta 和 best 记录；
- `UIDefectMetrics`：Base/checkpoint × 五类 × image/bbox 的 P/R/F1，以及 macro/micro 总指标。

缺失值保持空白，不写成 0。`openpyxl` 缺失或保存失败只产生 warning，不会终止训练；
JSON/JSONL 仍是 source of truth，Excel 可随时重建。

## 8. 验收

本地 CPT 单测：

```bash
python -m unittest \
  tests.test_cpt_split \
  tests.test_cpt_observability \
  tests.test_cpt_eval_metrics \
  tests.test_cpt_checkpoint_selection \
  tests.test_cpt_overfitting \
  tests.test_cpt_excel \
  tests.test_cpt_evaluator_end_to_end \
  tests.test_cpt_eval_queue \
  tests.test_cpt_smoke_validation \
  tests.test_locany_cpt \
  tests.test_locany_cpt_merlin
```

集群启动前由 A800 smoke 核对：20 step 正常、per-task sample/token/skip/CE 均存在、
resume 后计数单调不重复、多 rank reduce 不死锁、原始 TorchElastic 异常可见，以及统计
开启后 step time 增幅不超过 5%。这些结果必须来自真实 job 日志，不能由本地静态检查代替。

A800 smoke/eval 完成后自动核对 checkpoint-10→20 resume、rank state、对账、单调性、
十任务 CE、eval queue 和三表 Excel：

```bash
python scripts/validate_locany_cpt_smoke.py \
  --run a800=/path/to/a800-smoke-run \
  --require-eval \
  --eval-samples-per-task 10 \
  --output /path/to/cpt_smoke_validation.json
```

`--require-eval` 还会确认最终 checkpoint 的 queue 状态为 completed、十任务
`eval_token_ce` 非空、推理错误为 0、task-macro 完整；启用 Excel 校验时还要求
`CPTEvalMetrics` 表至少包含十任务加一行 macro，`CPTUIDefectMetrics` 至少包含 Base 和
checkpoint 的五类 image/bbox 行，而不只是存在空的 sheet。

常见失败含义：

- `Checkpoint completion ... OSError: [Errno 38] Function not implemented`：部分 ByteNAS
  挂载不实现 `fcntl.flock` 或 `fsync`。CPT eval queue 现在优先使用 `flock`，收到明确的
  `ENOSYS/ENOTSUP` 后退化到带超时和 stale recovery 的原子目录锁；marker/queue 的
  `fsync` 不支持时使用 close + 同目录原子替换。已经通过 resume 校验的 checkpoint 会在
  重启时补齐 completion marker 和 queue row，不会重训或静默删除。
- `ImportError: libGL.so.1`：OpenCV wheel 缺少任务容器的系统动态库，不是 CUDA/NCCL
  错误。所有 CPT train/eval YAML 默认设置 `INSTALL_SYSTEM_RUNTIME_DEPS=1`；launcher 会在
  `torchrun` 前预检，仅在确实缺失时通过 `sudo apt-get` 安装 `libgl1 libglib2.0-0`，随后
  再次验证 `import cv2`。在 master/login 节点单独安装不能修复下一次新建的 Merlin
  容器。若镜像已预装，预检直接通过且不会执行 apt。需要禁止自动安装时，A800 提交命令
  追加 `--no-install-system-runtime-deps`，或直接运行时设置
  `INSTALL_SYSTEM_RUNTIME_DEPS=0`。

  如需立即修复并验证**当前这个容器**，可逐条执行；新提交的 Merlin 容器仍由 launcher
  自动处理：

  ```bash
  sudo apt-get update
  sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libgl1 libglib2.0-0
  /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything/bin/python \
    -c 'import cv2; print(cv2.__version__, cv2.__file__)'
  ```

- `DummyOptim ... param_groups`：旧版 CPT 自定义 optimizer 路径；当前兼容层不再直接假定
  DummyOptim 暴露 `param_groups`。
- `UI modules ... no parameter update`：旧版将 UI5 专属断言用于 CPT；当前仅在 UI5 模块
  真正启用时检查。
- `immutable run configuration changed`：resume 的采样/world/token/数据配置与 checkpoint
  不一致，需恢复原配置或换新输出目录。
- `missing dataloader_state_rank...`：checkpoint 不完整，禁止静默重启数据流。
- oversize warning：查 `pre_mtp_already_oversize` 与 `mtp_expansion_oversize`，不要直接截断答案。

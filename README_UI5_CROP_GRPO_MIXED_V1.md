# UI5 Crop Mixed Rollout GRPO · H20 × 2

分支 `codex/ui5-crop-grpo-mixed-v1` 基于 `codex/ui5-crop-curriculum-v3` 的
`b29590f88c4b4a102c742f8410e1c6751b0859d2`。目标是原图级 UI5 image/bbox macro F1。
保留 boundary_v3、text-v3-1 监督修正、Detail Pyramid/relation/PBD 权重和正式 evaluator。

## 内网正式提交

把本分支完整 Git 提交导入内网后，在该提交的仓库根目录执行：

```bash
bash shell/submit_ui5_crop_grpo_mixed_v1.sh
```

这条命令依次完成独立 checkout、读取原提交的真实绑定配置、crop 四路选样和缓存复用、
新建 step 0 初始化、生成最终 YAML、调用 `mlx job submitv2 --path`。
集群任务随后运行新 step 0 完整评测、1200 步 GRPO、每 200 步完整评测、Excel 和 checkpoint 保存。
没有独立 debug 任务、指标早停或生成服务。

bootstrap 只依赖系统 Python 标准库，随后切换到原记录的 conda；不会安装或升级集群依赖。
读取的起点为：

```text
/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_logs/ui5_curriculum/locany-ui5-crop-rollout4-curriculum-hour021-h20x2-sdpa7268-20260906T073110Z-746aed
```

只沿该目录的持久化 restart/v3 successor 指针读取配置，不搜索“最新目录”。校验指针回链、
相同冻结截止点、YAML 与 runtime 一致。集群资源、镜像、卷、namespace、conda、processor、
rollout bundle、评测输入和 detector cache 都从这份真实记录继承。采用 boundary_v3/hybrid 正式评测。

正式提交可通过 `--cluster` 选择 H20 资源组和队列：

| 参数 | 资源组 ID | 队列 |
| --- | --- | --- |
| `--cluster default`（省略参数时相同） | 继承原 v3，当前为 1000 | 继承原 v3，当前为 `compute-329-hl-cloudnative-ai-iesqa.llm4se-guarantee` |
| `--cluster ies_aiai_experience` | 1602 | `compute-329-hl-cloudnative-ai-ies.aiai.experience-guarantee` |

`1602` 写入 Arnold 的 `groupIds`。物理 `clusterId` 沿用原 HL 配置（20），H20×2、CPU、内存、镜像、挂载
及训练配置继续继承。选择按每次提交生效，写入最终 YAML、该次 `submission.json` 和 `delivery_paths.json`
的 `submission_target`，不修改已有 `run.json` 的身份，可复用同一轮的 step 0 和完整 resume。
后续重提若继续使用新队列，也要携带 `--cluster ies_aiai_experience`。

新任务选择该目标：

```bash
bash shell/submit_ui5_crop_grpo_mixed_v1.sh --cluster ies_aiai_experience
```

已有任务更新代码后，选择该目标重提：

```bash
bash shell/submit_ui5_crop_grpo_mixed_v1.sh --resume-code-update --cluster ies_aiai_experience
```

新 checkout 固定为：

```text
/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Embodied-ui5-crop-grpo-mixed-v1
```

它从导入的 HEAD 创建独立 worktree，并绑定代码 SHA。已有目录必须是同一 SHA 且没有已跟踪文件修改。
不会改变旧 v3 checkout 或旧输出。默认 `RUN_NAME=ui5-crop-grpo-mixed-v1-h20x2-20260907`。
需要另一轮独立实验时，可显式传 `--run-name`；同一轮恢复使用以下命令，保留所有身份：

```bash
bash shell/submit_ui5_crop_grpo_mixed_v1.sh --resume
```

`--resume` 仅在原平台任务已经停止后使用。默认提交带持久化 attempt marker 和 MLX 回执，
重复执行不会重复提交；pipeline 另有进程锁。`--resume` 生成独立提交日志目录，训练输出和 reference 不变。

### 运行时兼容修复后重提（保留已完成的 step 0）

若原任务因 DeepSpeed `scale_wrt_gas` 签名检查、AR 概率校验或 ZeRO-2 通信顺序异常退出，
平台任务停止后执行：

```bash
cd /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Embodied-ui5-crop-grpo-mixed-v1
git pull --ff-only origin codex/ui5-crop-grpo-mixed-v1
bash shell/submit_ui5_crop_grpo_mixed_v1.sh --resume-code-update
```

该开关包含 `--resume`。仅接受运行时修复文件范围内、原提交后继的代码更新，记录原始和实际执行 SHA、
逐文件 Git blob 清单，写入 `runtime_code_revision.json` 及 `diagnostics/code_revisions/`。
原 `run.json`、mixed manifest、初始 reference、已保存的 optimizer/RNG/sampler 和评测身份不变。
已完成 hybrid step 0 会核验原指标哈希并复用；没有完整训练 checkpoint 时从 optimizer step 0 开始。
后续 checkpoint、运行环境诊断和 Excel 都记录实际执行的代码 SHA。

两个原生 AR 文件的数值/概率复算修复仅限没有已保存 optimizer 更新的任务；
已有 `resume/latest`、`.previous` 或 `.pending` 时首次绑定这些文件的变更会拒绝。
修复获绑定后可正常保存、恢复所有训练状态。原生成数值版本的固定 TRAIN AR step 0 诊断会归档到
`diagnostics/previous_train_ar/`，使用同一初始权重、同一固定 train 子集重算当前版本诊断，
避免迁移表混用数值路径；这不会重跑 hybrid 正式测试 step 0。

本次分段复算版本为 `prefill-tokenwise-layer-replay-v1`，仅改 current/reference 的计算调度；
生成数值版本仍为 `fp32-head-fixed-sdpa-v1`，所以已完成的同版本固定 TRAIN AR step 0 也可以复用。
第 4 步退出时尚未到默认的第 100 步保存节点；没有完整 `resume/latest` 时重新从 optimizer step 0 训练，
不会把未保存的第 1–3 步当成可恢复 checkpoint。

DeepSpeed 0.16/0.17 的 [NVTX 装饰器](https://github.com/deepspeedai/DeepSpeed/blob/v0.17.5/deepspeed/utils/nvtx.py)
可能只暴露 `(*args, **kwargs)`；不能据此断定不支持
`scale_wrt_gas`。兼容检查接受关键字转发，实际调用仍为 `backward(loss, scale_wrt_gas=False)`，
不乘 GAS 补偿、不重复调用 backward，也不升级继承环境。

## 产物路径

最终平台 YAML 在内网读取真实记录后生成，不能从外部开发机猜填继承配置：

```text
/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_logs/ui5_grpo/ui5-crop-grpo-mixed-v1-h20x2-20260907/formal.yaml
```

仓库内 `jobs/ui5_crop_grpo_mixed_v1_h20x2.yaml` 是渲染入口，渲染器保留实际资源配置，
写入确定的代码 SHA、conda、所有运行路径及完整 `GRPO_CONFIG_JSON`。通过上面的单条命令提交，
不要绕过选样、初始化和绑定过程直接提交未渲染的仓库 YAML。

输出根目录为：

```text
/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_models/Embodied-ui5-crop-grpo-mixed/ui5-crop-grpo-mixed-v1-h20x2-20260907
```

以下相对路径均位于该输出根目录：

| 路径 | 内容 |
| --- | --- |
| `run.json` | 不可变运行配置、代码 SHA、继承配置、数据和固定 reference 身份 |
| `delivery_paths.json` | 最终 YAML、Excel、manifest 和 checkpoint 的完整绝对路径 |
| `mixed/manifest.json` | 真实候选数量、四路证据、排除原因计数、任务/正负/k strata、空 strata、计划比例及 PNG 复用量 |
| `mixed/groups.jsonl` | 唯一 RL 题池，以原图-task 为单位，包括全部 crop 计划和当前 GT |
| `mixed/replay.jsonl` | 原始可用训练全集的 source-balanced 回放记录，四个局部任务为 crop，content_missing 为整图 |
| `mixed/raw_evidence.jsonl` | 冻结前可见的 crop 原始记录路径、行位置、逐记录 SHA |
| `mixed/train_diagnostic.jsonl` | 固定 mixed train 诊断子集，每任务至多 12 个 group |
| `initial_model/` | 原 checkpoint 的独立元数据/代码和只读硬链接权重，无旧 optimizer/scheduler/step |
| `resume/latest/` | 完整滚动模型、ZeRO-2 optimizer、scheduler、各 rank RNG/sampler 和 global step |
| `checkpoints/best-image` | 最佳 image macro F1 的模型目录链接 |
| `checkpoints/best-bbox` | 最佳 bbox macro F1 的模型目录链接 |
| `checkpoints/best-joint` | 现有 v3 joint 规则选出的模型目录链接 |
| `evaluation/step-000000/` 等 | 完整测试预测、原始答案、worker 汇总、正式指标和输入身份 |
| `diagnostics/ui5_grpo_training_evaluation.xlsx` | 正式训练和评测 Excel |
| `diagnostics/sampling_step000100.json` 等 | 累计实际采样比例，续训连续累计 |
| `diagnostics/train_ar_step000000.json` 等 | 明确标为 TRAIN/slow 的四次正确数及相对 AR step 0 的迁移 |
| `metrics/step000010.json` 等 | 每 10 optimizer steps 的真实训练统计 |
| `trajectories/step000001/rank0.pt` 等 | CPU 采样 token、detached old/reference log-prob、prompt ID/position/mask、原始输出及奖励 |

原初始化 checkpoint 固定为：

```text
/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_models/Embodied-ui5-det-crop/locany-ui5-v5-croponly-sourcebalanced-a800x4-20260830/checkpoint-12000
```

首次训练从 step 0 新建 optimizer/scheduler。reference 永远指向这份初始权重身份，
不会随 actor、best 或 resume 改变。step 0 若仍为最佳，best 链接指向 `initial_model`。
每 100 步保存完整滚动状态；只保留当前三个 best 对应的副本，删除已被替代的本轮副本。

## 选样与监督

旧 rollout 根目录固定为：

```text
/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_rollouts/ui5-train-rollout8-h20x2-v6-20260904
```

`build_ui5_grpo_mixed_manifest.py` 验证实际绑定的 hour021 snapshot 是 frozen selection 的来源。
同时读取 snapshot 的 `complete8.jsonl` 和 `incomplete_or_technical_error.jsonl`，按原始 crop payload
逐条相等核对四路 raw。这个边界采用冻结时真正读到的记录；单纯用文件 mtime 或时间戳会把后续记录混进来。

只要求 crop 自己的四次结果解析有效、无 runtime/坐标/标注/覆盖异常。独立核对所有 crop 的集合、
坐标变换与原图合并结果，使用现有匹配器在 IoU=0.1 重新计算原图 exact_correct，选 1/4、2/4、3/4。
m31 的正确数、完整性或失败不参与资格条件；旧 complete8 仅是冻结证据的一部分。
不把 defect 有无判断、四个图块或历史答案当成四条训练轨迹。

原图内容和 task 去重，沿用现有内容哈希做完整 train/test 隔离。先在有候选的任务间平衡，再在任务内
平衡可用原图正负，最后平衡可用 k=1/2/3；各层使用固定 seed 的无放回循环。空 strata 明确记录，不合成候选。
真实数量由内网数据统计，仓库不预填数量。

局部 crop 计划来自 GT-free `base_scan_plans.base_tiles`，复用 detector_scan 坐标变换和已有 PNG。
仅为当前 mixed 集补缺失 PNG，其他回放 PNG 必须已有。回放合并原来三个池，它们构成完整可用训练全集，
而不是仅使用旧 `global_replay` 池。当前真实 GT 重新生成监督：负例严格 `<box>none</box>`；
正例一个既有英文 ref 加全部有效框；真实 chat template 添加 EOS。SFT 回放禁止截断答案，
保留原 AR/MTP、image gate、slot gate、attention 等既定辅助 loss。

## 原生在线训练与概率

每个原图-task group 从固定的当前 actor 采样 G=4。每条轨迹访问同一组全部 crops，采用独立随机流，
回映射后 NMS=0.5 得到一份原图预测。content_missing 每条轨迹只看一次整图。
同一轨迹的原图奖励分配给它的所有 crop completion。

`eaglevl/model/locany/ui5_ar.py` 共享原生视觉、Detail Pyramid、任务/relation、PBD 和 Qwen2 SDPA。
采用 `generation_mode=slow` 的逐 token AR 数学路径，保留真实 multinomial 采样 token；
不使用 hybrid `decode_bbox_avg` 的重写文本计算 AR 概率。slow 的 PBD 在 `<box>` anchor 位置生效，
复算采用相同位置，不把字面采样的 `<text_mask>` 当成 MTP block。

采样 temperature=0.7、top_p=1、top_k=0、repetition_penalty=1。old/current/reference 都是
完整词表 logits / 0.7 的同一分布。old log-prob detached 后保存到 CPU，无常驻 old actor。
三者统一在 **FP32 中执行词表投影和 log-softmax**，避免先产生 BF16 logits 再转 FP32 的舍入损失；
模型权重、LLM 主干和视觉仍为 BF16。关闭 TF32 matmul 和 BF16 GEMM 的低精度中间归约，
不修改权重值，不用整段复算值替换真实采样 old log-prob。
只取真实 completion 对应的 shifted logits/log-prob，包含实际 EOS；prompt、视觉占位及补齐位置不计损失。
达到长度预算时不伪造 EOS。生成与复算都为 eval 模式，关闭 dropout；只有 current/replay 复算开启梯度。

每个 completion 微批复算全部输入，不复用跨权重更新的视觉或 prefix 图。固定权重生成阶段内，
同一 view 的视觉特征与 prefix KV 在四条轨迹间复用。当前策略前向返回 completion log-prob，
普通标量 CE 只用于原 SFT 回放。

current/reference 按采样时的 **完整 prompt prefill + 每次一个答案 token** 复算，
不再把 prompt 和整条 completion 一次送入 decoder。每层的 Q/K/V、SDPA、MLP，以及末尾 norm、
PBD 和 FP32 词表投影都使用相同分段形状；最后一个采样 token（包括真实 EOS）作为 target 计分，
不额外运行一个采样阶段不存在的 decoder step。采样概率仍直接来自 multinomial 的真实分布。

利用因果依赖，复算按层处理该层的全部 token，调用原生 decoder，并采用按层及按 token 的
非重入 checkpoint。每次重算重新创建局部 KV 容器，避免重复追加/读到未来 token；
KV 张量保留梯度，末尾 token 的损失可以传回全部可见前缀与视觉/任务模块。
不把生成阶段的 detached KV 传给训练；不同时保存所有层、所有 token 的增长 KV 历史和展开后的
GQA attention 激活。逐 token 复算增加前向调用次数，吞吐与显存以正式任务日志实测为准。

仅原生 GRPO AR 的 decoder 固定 SDPA 后端：H20 为 PyTorch `EFFICIENT_ATTENTION`，CPU 验证为 `MATH`。
后端上下文位于每个 checkpointed layer 的 callable 内，覆盖生成、reference、current 和 backward
重算；hybrid 与 SFT/MTP 的 decoder 分支保持原实现。长序列 mask 只填充底层存储以满足 CUTLASS
行 stride 的 8 元素对齐，再切回原 shape，不添加输入 token、不改 causal/window 可见性。
不在 H20 上强制 materialize 完整 FP32 attention 矩阵。

仅统一输出层精度和 SDPA 后端不足以消除整段/单 token 的计算差异。
[PyTorch 数值说明](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)指出，批量与切片计算
不保证逐 bit 相同。本次复算直接对齐分段形状，并保留原有 ratio 与数值 KL 检查，不放宽阈值、
不重写 old log-prob。`runtime_environment.json`、轨迹及失配证据同时记录生成与复算版本；
逐卡 `diagnostics/progress/rank*.json` 在每次完整更新后保留本步概率一致性摘要。

奖励使用正式 parser、IoU=0.1 匹配及原图集合：

```text
F_box = 2 TP / (2 TP + FP + FN)
C_img = 1[预测与 GT 有无缺陷一致]
S_iou = 2 sum(assigned IoU) / (|P| + |B|)
R     = .70 F_box + .20 C_img + .10 S_iou
A_i   = R_i - mean(R_1, R_2, R_3, R_4)
```

S_iou 包括匹配器分配的低于 0.1 的 IoU。双方空时三项为 1，仅一方空时三项为 0。
任一 crop 为正式 parser invalid 时整条轨迹 R=-1，原始输出和该轨迹仍保留。
相同四个奖励产生零策略优势，继续 KL 和 replay，不使用标准差归一化或零优势丢组。

## H20 × 2 配置及同步

| 设置 | 正式默认值 |
| --- | --- |
| 精度 / Attention / 分片 | BF16 / LLM SDPA / ZeRO-2，开启 gradient checkpointing |
| 参数 | 冻结 vision backbone，其余沿 v3 分组并保留全部初始权重 |
| 每 optimizer step | 每卡 2 轮 × 每轮 1 group，共 4 groups / 16 原图轨迹；每 group 1 条 SFT replay |
| 预算 / LR | 1200 steps；LLM 1e-6，warmup 40，cosine 到 5e-7；其余组按 v3 相对倍率 |
| GRPO | clip=.2，KL beta=.02，num_iterations=1，max_grad_norm=1 |
| 长度 | 三个序列/token 上限均 7268；输出 ≤512，按 worker 的剩余预算规则截限 |
| 总 loss | L_policy + .02 L_KL + .10 L_SFT_replay |

生成、reference 打分、current/replay 反传分阶段执行。reference 平时驻 CPU，打分阶段移入当前 rank GPU，
完成后返回 CPU；轨迹始终存 CPU，逐 completion/crop 反传。

策略和 KL 先除以各 group 的四条完整轨迹的 completion token 总数，再等权平均 groups。
每 rank 的两个 group 除以 2，ZeRO 的跨 rank 平均再除以 2；crop 多的长图不增加 group 权重。
变长 crop 数以跨 rank MAX 对齐微批槽，空槽使用全参数零权重依赖，避免不同参数参与或不同步通信。
显式指定最后槽为唯一 optimizer boundary，关闭 DeepSpeed 的额外 GAS 缩放；GAS=2 表示两轮 group，
不解释为两个 crop。每步核对 engine.global_steps，每个 completion 在更新前核对采样/复算 log-prob
及有限值/长度。更新前所有 `exp(current_logp-old_logp)` 必须位于 `[1-clip, 1+clip]`（本轮 `[0.8, 1.2]`），
且 `mean(expm1(delta)-delta) ≤ 0.001`，其中 delta=current-old；数值误差不得提前触发 PPO clipping，
也不能有明显的整体分布漂移。后者是采样/复算数值诊断，与固定 reference 的 KL loss 分开记录。
原 max/mean log-prob 误差继续记录，并加入 min/max ratio、sampling_numerical_kl；未通过仍终止技术错误，
不丢 crop、不改 old log-prob、不作为模型难度或指标早停。

CPU 轨迹在反传前保存；超限时 `diagnostics/ar_mismatch/stepXXXXXX-rankR-slotS.pt` 额外保留真实 tokens、
raw output、old/current/reference log-prob、输入 ID/位置/mask、最大误差 token 索引、group、代码和数值版本。
这些检查与记录内嵌正式训练，不创建额外 debug 任务。

### ZeRO-2 固定梯度归约顺序

仅对齐槽数和全参数零依赖不能固定 autograd hook 的到达顺序。原生
[DeepSpeed ZeRO-2](https://github.com/deepspeedai/DeepSpeed/blob/v0.17.5/deepspeed/runtime/zero/stage_1_and_2.py)
在 hook 到达时填充 IPG bucket；两卡分别执行 AR、MTP replay 或 padding 时，bucket 的参数组合和
collective 数量可能不同，即使两边参数总数相同。

`ui5_grpo_zero2.py` 在梯度就绪与原生 reducer 之间加入固定顺序队列。两卡从相同 optimizer 参数组
生成同一个反向参数顺序，先核对参数名、shape、dtype、分组和 bucket 配置的 SHA。
每个槽只缓存就绪参数的引用，按该顺序交回原生 reducer；所有参数完成后才允许进入原生 reduction
epilogue 与 optimizer step。缺失/重复梯度立即报错。ZeRO stage=2、原生分片/累积/归约缩放、
overflow、梯度裁剪、Adam 和 scheduler 都保留，不增加 optimizer step 或 GAS 缩放。

固定顺序可能延长部分 `.grad` 的驻留，但不复制梯度、不保留额外 completion 计算图；每 10 steps
记录 `max_pending_gradient_elements` 与原有显存峰值。每卡每槽仍为一条 completion/crop 微批。
这属于运行时通信修复，可以用 `--resume-code-update` 接续同一 run；当前 AR 数值版本与两个 step-0
基线均不变，无需重算已完成的 hybrid 评测或当前版本的固定 TRAIN AR 诊断。

正式任务内保存 `diagnostics/zero2_reduction_plan.json`、`trajectories/stepXXXXXX/rankR-schedule.json`、
以及持续更新的 `diagnostics/progress/rankR.json`，标明 generation/reference/具体微批槽/optimizer 完成。
启动首个 optimizer update 成功时打印 `[GRPO UPDATE]`。训练子进程默认启用 NCCL flight recorder
（8192 events）、超时 dump 和 desync 信息，产物位于 `diagnostics/nccl/`；不延长超时、不修改网络传输，
不另起 debug job。变量含义见 [PyTorch NCCL 文档](https://docs.pytorch.org/docs/2.14/torch_nccl_environment_variables.html)。

## 评测、Excel 与完整恢复

step 0、200、400、600、800、1000、1200 全量 UI5，始终 boundary_v3 + hybrid。
训练子进程退出释放 GPU 后，复用正式五任务 worker 和 evaluator：GPU0 为 occlusion/cropping，
GPU1 为 text_overflow/text_ellipsis/content_missing。评测通过完整输入/输出覆盖和 invalid 校验后才登记。
测试输入 JSONL、原图内容及 detector manifest 固定哈希，所有节点一致。测试集不进入采样或训练奖励。

每 10 steps 记录未加权 policy/KL/replay、总 loss、三项奖励、非零优势组比例、invalid、
有效/总 group 与轨迹数、token、吞吐、生成/reference/反传时间和显存峰值。统计是 optimizer step 口径。
每 100 steps 更新 Excel，每个完整评测节点也更新。五任务与 macro/micro 的 image/bbox P/R/F1、
混淆计数、invalid 和相对 step 0 增量都保留；最后两张表固定为 `image_detail`、`bbox_detail`。
BBox TN 按既有 evaluator 的可用性留空，不填虚构的 0。

固定 mixed TRAIN 子集额外记录 slow AR 四次正确数迁移；它与主 hybrid 测试结果分开标注。
主 checkpoint 选择完全复用 v3 best-image/best-bbox/best-joint 规则。

完整 rolling checkpoint 使用 `.pending → latest` 事务、SHA/文件清单和各 rank 状态。
中断后仅接受完整 marker，恢复模型、optimizer、scheduler、RNG、group/replay sampler 位置、
累计采样计数、global step 和固定 reference 身份。未提交的未来日志重算；已完成同一身份的评测不重复计分。

## 验证

普通单元/集成测试，不创建集群 debug job：

```bash
python -m pytest tests/test_ui5_grpo.py tests/test_ui5_grpo_native_ar.py tests/test_ui5_grpo_distributed.py tests/test_ui5_grpo_zero2.py tests/test_ui5_grpo_pipeline.py tests/test_ui5_grpo_manifest_integration.py tests/test_ui5_grpo_runtime.py -q
```

覆盖真实小型原生 Qwen2 SDPA/PBD 的 cached slow 与 teacher-forced 概率、原 slow decoder 等价性、
实际 EOS 和特殊 token、视觉/relation/PBD 梯度、官方奖励/低 IoU/invalid、冻结边界及 m31 缺失资格、
真实 PNG 复用补缺、两个真实 Gloo rank 的变长 group 梯度更新、RNG/sampler 恢复、best 副本事务和完整 Excel 导出。
Windows 单元测试只替换 POSIX best symlink 创建原语，副本/哈希/删除/幂等仍执行真实文件操作。

新增 BF16 大 logits（约 20–30）及多个 `<box>` 的概率/梯度回归、autocast 下的 FP32 词表投影、
不同外部 SDPA 默认下的 backward checkpoint 重算、7268 等非 8 倍数长度的 mask 存储对齐，
以及观测误差量级的接受条件、真正 ratio 越界/整体漂移/NaN/Inf 拒绝条件和数值版本恢复审计。

分段复算回归逐层比较采样、前向复算、backward replay 的输入值、形状、位置和 causal/window mask；
旧提交在该形状对照中失败，新实现通过。另与独立的逐 token、保留完整 KV 计算图的梯度 oracle 比较，
验证末尾 token 对 prompt/视觉模块的梯度；检查外层 checkpoint 不保存所有层的 4-D KV/attention
历史，并执行 BF16 模型连续 5 次权重更新后的真实采样概率检查。

`test_ui5_grpo_zero2.py` 直接运行 DeepSpeed 原生 ZeRO-2 的 leaf hooks、IPG buckets、梯度分片、Adam
及 optimizer state 保存/恢复，用两个 CPU/Gloo rank 与集中计算的 group/token 归一化更新比较。
测试主动置换两卡的合法梯度就绪通知顺序：旧实现出现 bucket 参数/顺序差异，固定顺序后保持一致；
同时覆盖不同 crop 数、零优势组、replay、空槽、多参数组和超出 bucket 大小的参数。
CPU 测试仅禁用可选 shared-memory JIT extension，通信执行真实 Gloo；没有 DeepSpeed 时显式 skip，
不能把这种 skip 计作 ZeRO-2 验证通过。

联合测试进一步把真实小型 Qwen2/PBD 的 BF16 分段 AR 复算接到原生 ZeRO-2 上，
两个 rank 各自完成 G=4、不同 crop 数、不同监督图及空槽，连续更新 3 次，核验 on-policy 概率和
更新后的两卡权重一致。测试监督分支只验证 hook/通信组合，完整 UI5 SFT/MTP 保持原测试覆盖。

已完成本次分段复算修复后的训练相关完整回归（110 项测试及 2 个子测试，包含双 rank、提交/恢复和 Excel），
其中原生 ZeRO-2 专项测试在 DeepSpeed 0.17.5 与 0.16.3 的两个 CPU/Gloo rank 上均通过，未跳过。
shell 语法与 Git diff 空白检查通过。Excel 测试产物完成数值回读和渲染检查。

外部开发机没有内网挂载或 H20，测试不代表真实训练已执行，也不预填真实样本量、F1 或吞吐。
内网单条正式命令生成真实 manifest/YAML 并运行上述完整训练流程，概率与同步检查嵌在正常训练内。

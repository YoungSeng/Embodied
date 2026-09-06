# UI5 Crop Curriculum v3 — H20×2

分支：`codex/ui5-crop-curriculum-v3`。这是新的实验，不保证 F1 提升。
仅 UI5；复用 hour021 已发布的 frozen selection、课程分组和 PNG。
当前版本先发布独立的 `text-v3-1` 训练文本/recipe/manifest，再提交新运行。
不重跑 rollout、不调用课程 builder、不重新生成或复制 crop PNG、不修改旧运行或旧文本。

## 负例监督修正：text-v3-1

所有池统一按当前训练视图的 GT 判断正负，不能用原图是否有缺陷代替：

- crop 使用 `_ui5_crop_gt_local_1000`；content_missing 整图使用 `_ui5_union_gt_1000`。
- 空 GT 的完整答案严格为 `<box>none</box>`，不含 `<ref>`、空白、EOS 或额外标签。
- 非空 GT 为 `<ref>该任务的原有英文标签</ref>` 后接全部 norm1000 框，保持 GT 原顺序。
- 数据 JSONL 不手写 EOS；真实训练 chat template 添加 `<|im_end|>`。

修复了 builder 将原图 `<ref>` 无条件带到无缺陷 crop 的逻辑；三个池和整图走同一格式化规则。
正式入口还会读取已有 JSONL，生成独立版本，而不是只更新 builder 后继续使用旧文本。
只改需要修正的 assistant 答案：样本 ID、GT、样本数、prompt、图片字段完全保留；
已经正确的 JSONL 行按原字节复制。无法核实独立 GT 时直接报错，不从答案猜 GT，不填补坐标。
此修正不改变评测 parser、invalid 处罚、解码器、loss 权重或课程比例。

默认的新训练文本目录（`<hash12>` 绑定旧发布身份及修正实现，入口会打印完整路径）：

```text
/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_data/ui5_curriculum/
  hour021-s42-reuse-20260905T065342Z-f8d36a-text-v3-1-<hash12>/
    hard.jsonl
    matched_anchor.jsonl
    global_replay.jsonl
    ui5_crop_rollout4_curriculum.json
    curriculum_manifest.json
    supervision_format.json
    _SUCCESS.json
```

recipe 的三条 annotation 都指向上面新 JSONL 的绝对路径。
PNG 不搬迁、不遍历生成，新 manifest 的 `crop_asset_root` 继续指向原图片目录。
冻结 hard/anchor 分组文件按原字节保留；新 manifest/SUCCESS、正式 YAML、训练读取及 checkpoint
连续性身份绑定新 recipe 和文本 SHA256。改回旧 annotation、篡改文本或使用旧 manifest 都会被拒绝。
相同输入/实现再次准备可校验后复用已经发布的新文本；不覆盖未完成的发布目录。

`supervision_audit()` 逐条比较完整答案与当前视图 GT，包括任务 ref、全部坐标、顺序及尾部内容，
不再只数标签。`supervision_format.json` 分池记录 ref 负例修正前/后数量、总修改数、
原图正→crop 负数量、样例 sample/crop ID、修改前后文本、新旧路径和 SHA。
正式训练构造真实 dataset 后，每池选定第一个实际负例，执行同一个 processor、图像预处理、
chat template、tokenizer、截断和 `get_targets_flag_with_mtp`，检查：

```text
AR:         <box> none </box> EOS
MTP block1: <box> none </box> <null> <null> <null>
MTP block2: EOS   <null> <null> <null> <null> <null>
```

审计不推进 sampler，完整恢复 Python/NumPy/Torch/CUDA RNG，实际 token ID 写入
`diagnostics/negative_mtp_audit_rank*.json` 并打印 `[NEGATIVE MTP PASS]`。
新全量 step 0 发生在训练前，因此该节点的实际 MTP 审计尚不可用；从 step 200 起必须有真实报告，
不会把 CPU mock 测试结果冒充 H20 tokenizer 实测。

## 已核实的旧运行证据

证据包 `ui5_step0_step200_evidence.tar.gz` SHA256：
`7c7f3bb61522d9819a6150af4c6dd50f7f31242a9c228d6393523f1f4cc045e9`。
实际旧运行：`locany-ui5-crop-rollout4-curriculum-hour021-h20x2-sdpa7268-20260906T073110Z-746aed`。

| 指标 | step 0 | step 200 |
| --- | ---: | ---: |
| image macro F1 | 0.53271068 | 0.16979986 |
| bbox macro F1 | 0.43902289 | 0.18829424 |
| image macro recall | 0.55382433 | 0.55567728 |
| bbox macro recall | 0.40875645 | 0.41621643 |
| invalid images | 1 | 2426 |
| worker runtime errors | 0 | 0 |

导出包不是全量 raw，不能把未导出的正常图片计为 runtime/missing。
但各任务的 invalid raw 数与 worker summary 完全一致，覆盖全部 2426 个失败图像。
其中失败 tile 共 2921：ref/box 标签混接 2819，ref 未闭合 99，box 未闭合 3。
例如 `step-000200/ui_text_overflow/raw/154884_parse_error.json` 的 tile 2 是
`<ref>none</box><|im_end|>`，前两个 tile 均为正常 `<box>none</box><|im_end|>`。
`step-000000/ui_cropping/raw/153839_parse_error.json` 已出现一次 `<ref>cro</box><|im_end|>`。
这不是把 invalid 转成空负例就能合法解决的问题。

定位到两个具体解码缺口：旧 `decode_ref` 接受任意非坐标 future-slot token，
可以把并行预测拼成上述混接文本；旧 hybrid AR 又把除坐标/none/box_end 外的 token 当作终止。
v3 的 `boundary_v3` 在 ref/text 边界只提交实际采样的首 token，再逐 token AR；
遇到真实 `</ref>`/`</box>` 才回 MTP，只有真实 EOS 终止。
MTP 首位 null 不再被改写为 EOS；AR 的 null/普通文本保留在 raw，继续生成。
不修补已有 raw，不合成缺失坐标或闭合标签，不改变 parser、IoU、匹配、tile 合并或 invalid 处罚。
原有 bbox 专用解码与跨 crop NMS 保留。

历史 raw 没有逐 token trace，包中也没有 tokenizer/模型文件：
不能据此断言长度截断或 token ID 错配是实际原因，也不能证明训练为何放大了该模式。
本地 CPU 测试复现了上述解码分支，并直接检查现有 MTP 负例标签构造：
`<box>none</box>` 后是 null padding，独立下一块监督 EOS；没有把二者混为一谈。
真实 tokenizer 与 processor 的特殊 token/1001 个坐标 ID 在训练、推理启动时逐项核验并留档。

**GPU 配对实验尚未在本地运行，修复后的实测收益以新任务报告为准。**
正式入口会对固定 hash 选取的每任务 32 张 UI5 样本、seed 42，运行：
原模型/可用退化模型 × legacy hybrid、boundary_v3 hybrid、boundary_v3 slow。
它们复用正式五 worker、裁剪、坐标回映射、NMS 和 evaluator，不建立另一套推理实现。
legacy 仅用于同权重配对测量；正式全量 step 0 和所有训练后评测都固定为 boundary_v3 hybrid。
退化权重只作比较，不进入训练；若 step-200 权重已被滚动覆盖/删除，则明确记录 unavailable，
不会拿后续 step 的权重冒充 step 200。可用 `--degraded-checkpoint /absolute/path` 显式指定。
不根据比较结果自动切换解码策略，也不以 F1 设停训条件。

## 提交：开发机当前窗口一次执行

先确认旧训练已结束，不要在旧任务仍运行时切换其共享代码 checkout。
以下命令只提交一次新的 text-v3-1 任务；保留所有旧提交记录，不删除任何 restart reservation。
新的 `curriculum-v3-text-v3-1.started` 与之前的 `curriculum-v3.started` 独立，
已经提交过旧 v3 不妨碍本次文本修正版；同一修正版重复提交仍会被阻止。

```bash
bash <<'BASH'
set -Eeuo pipefail
WORKSPACE=/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace
cd "$WORKSPACE/code/Embodied-ui5-curriculum"
BRANCH=codex/ui5-crop-curriculum-v3
git fetch origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
git switch "$BRANCH" || git switch --no-track -c "$BRANCH" "origin/$BRANCH"
git merge --ff-only "origin/$BRANCH"
CUDA_VISIBLE_DEVICES="" "$WORKSPACE/conda_envs/LocateAnything/bin/python" -u \
  scripts/ui5_curriculum_v3.py \
  --previous-submission-dir "$WORKSPACE/gui_logs/ui5_curriculum/locany-ui5-crop-rollout4-curriculum-hour021-h20x2-sdpa7268-20260906T073110Z-746aed" \
  --submit
BASH
```

CPU 准备阶段读取旧 `snapshot-switch.json`/`formal.yaml`，检查二者一致，
原样继承真实资源、processor、batch/梯度累积和目录配置。
metadata/hash 检查后扫描已有训练文本及旧 raw，发布答案修正版本，不解码训练图片，不生成 PNG。
准备阶段打印 `[TEXT PROGRESS]` 和各池 `[TEXT FIX]`，只有文本发布、完整格式审计通过才提交。
为原模型建立新目录的私有代码/config/tokenizer 副本及同挂载硬链接权重，
不会复制 optimizer/scheduler/trainer_state 或写入原模型目录。
原模型固定为：
`$WORKSPACE/gui_models/Embodied-ui5-det-crop/locany-ui5-v5-croponly-sourcebalanced-a800x4-20260830/checkpoint-12000`。
这是新训练的 step 0；原 crop checkpoint 的命名 12000 不是本次 global step。
若跨挂载无法硬链接或空间探测失败则报错，不隐式复制几十 GB 权重。

资源模板是 `jobs/ui5_crop_curriculum_v3_h20x2.yaml`；不能直接提交未绑定模板。
上面的命令自动生成绑定现有数据、代码 SHA 和独立输出目录的 **正式 `formal.yaml`**，
然后通过 `mlx job submitv2` 提交并核验平台返回。命令不启动后台准备进程。
省略 `--submit` 可只做 CPU 检查并生成 YAML；再次运行准备命令会产生另一独立输出目录。

## 训练及指标

| optimizer steps | hard | anchor | global replay | LLM LR |
| --- | ---: | ---: | ---: | ---: |
| 1–400 | 20% | 20% | 60% | 1e-6 |
| 401–800 | 25% | 25% | 50% | 7e-7 |
| 801–1200 | 30% | 30% | 40% | 5e-7 |

比例为 group draw 概率：先 pool、再 sample group、再组内轮转 crop。
hard/anchor 数来自已发布 manifest，不写死。五任务及池级正负覆盖均核验，
各 task/polarity 的真实 group 数（包括 0）列入 data_coverage；global replay 核验全部十个 strata。
不为补齐 frozen hard/anchor 中某个细分空格而制造/更换样本。
SDPA，三个 token/sequence 限制均 7268，batch 沿用单卡 1、梯度累积继承旧运行。
所有架构、训练 loss 权重和 LR 保留；在既有 v3 配比基础上修正负例监督格式。

流程：配对解码比较 → 新全量 step 0 → DDP 训练 200 → 退出 DDP →
GPU0 同时 occlusion/cropping，GPU1 同时 text_overflow/text_ellipsis/content_missing →
五进程退出后合并评分 → 更新 Excel/严格 best → 完整 resume/latest 续训。
每个 worker 在同一次加载中完成自己任务的 hard rollout4 和固定 anchor 推理。
anchor_retention 是相对新 step 0 的真实预测保持率，不再是名单交集。
hard 转移明确标为 train/mining、相对冻结 0/4；不能当作 held-out UI5 改善。
最终全量评测节点为 0/200/400/600/800/1000/1200，无指标早停。

`resume/latest` 保留完整模型/optimizer/scheduler/scaler（适用时）/RNG/sampler/global-step/阶段身份。
训练进程间恢复全部状态；三个 best 指针可共指一个严格创新高的永久 checkpoint。
不另存 400/800/1200 milestone，不从退化旧运行继承 optimizer。
操作失败会显式停止，不伪造成功指标或绕过完整性检查。

每次 `[V3 READY]` 打印实际 RUN_NAME、OUTPUT_DIR、正式 YAML 路径。

```text
$WORKSPACE/gui_models/Embodied-ui5-det-crop/ui5-crop-curriculum-v3-text-v3-1-h20x2-<UTC>-<unique>/
  diagnostics/v3_preparation.json                 数据/旧 raw/负例监督/存储检查
  diagnostics/decoder_comparison.json             同权重旧/新解码收益、hybrid/slow
  diagnostics/decoder_comparison/*/               比较 raw、worker 日志及正式 scorer 输出
  diagnostics/training_token_contract_rank*.json  训练 token ID 核验
  diagnostics/negative_mtp_audit_rank*.json         实际新 dataset/processor/MTP 负例标签
  diagnostics/ui5_crop_rollout4_curriculum_evaluation.xlsx
  evaluation/step-000000/... → step-001200/...     全量 UI5 raw/指标/anchor/hard
  checkpoints.json                               所有节点及 best 指针
  resume/latest/                                 精确续训状态
  logs/v3-start.log                               配对比较及启动日志
  logs/curriculum-*.log                           训练—评测循环日志
$WORKSPACE/gui_logs/ui5_curriculum/<RUN_NAME>/formal.yaml
```

Excel 包含 train_curve、ui5_overall、ui5_by_task、hard_transition、anchor_retention、checkpoints，
以及 data_coverage、output_validity、provenance、decoder_comparison、inference_fix_gain、training_vs_baseline、
supervision_format、training_text_identity、negative_mtp_supervision。
记录实际采样累计/窗口比例、实际 loss 分量（缺失为 null/空白，不填 0）、总体和五任务混淆计数、
invalid image/tile 分类、代码 SHA、数据 hash、解码配置及推理耗时。
bbox TN 沿用 evaluator 定义：不可用则空白，不发明目标框级 true-negative 总数。
进程日志打印训练状态/ETA、各任务进度、invalid 分类、各项 F1、best 变化及续训节点。

## 不占 GPU 的回归与证据复核

```bash
WORKSPACE=/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace
cd "$WORKSPACE/code/Embodied-ui5-curriculum"
CUDA_VISIBLE_DEVICES="" UI5_CURRICULUM_PROFILE=scheduled_v2 "$WORKSPACE/conda_envs/LocateAnything/bin/python" -B -m unittest -b \
  tests.test_ui5_curriculum_v3 tests.test_ui5_curriculum_evaluation \
  tests.test_ui5_curriculum_pipeline tests.test_ui5_curriculum_artifacts \
  tests.test_ui5_supervision_revision
```

证据包只读复核（不解压、不执行包内内容、不改变预测）：

```bash
python scripts/ui5_output_validity.py --evidence-archive /absolute/path/ui5_step0_step200_evidence.tar.gz \
  --output /absolute/new/path/evidence-audit.json
```

本地 CPU 回归不能替代 H20 上的真实模型加载、六组配对比较及 1200-step 实跑。

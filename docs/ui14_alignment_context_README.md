# UI14 alignment context

基于 neg11 `0379209bd6089e4deeed473d3108f1f41c8d97bf`，
分支 `codex/ui14-alignment-context-v1`。本次改动为原预测审计、生成格式约束、
独立补评和正式训练入口；ui_alignment 仍为全图输入，任务编号、模型结构与训练损失不变。

默认只读输入：

- neg11 数据：`/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_data/ui14_cpt9000_neg11_v1`
- 历史 run：`/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_models/locany-m32-cpt9000-ui14-a800x4-repair-v2`
- 历史 test：`/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_data/ui14_cpt9000_repair_v2/evaluation_manifest.json`

新目录：

- 代码：`/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui14-alignment-context`
- 清单/审计/补评：`/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_data/ui14_alignment_context_v1`
- 训练：`/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_models/locany-m32-cpt9000-ui14-alignment-context-a800x4-v1`

## 执行顺序

保持正在运行的旧 checkout 和训练不动。在开发机另建 checkout：

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4
git clone --branch codex/ui14-alignment-context-v1 --single-branch \
  https://github.com/YoungSeng/Embodied.git Embodied-ui14-alignment-context
cd Embodied-ui14-alignment-context
```

入口默认使用 `/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything/bin/python`。
本入口使用专属 `UI14_ALIGNMENT_DATA_ROOT` 和 `UI14_ALIGNMENT_PARENT_DATA_ROOT`；
旧实验残留的 `UI14_DATA_ROOT`、`UI14_PARENT_DATA_ROOT` 不会决定本次读写目录。
每次启动会打印最终路径；显式 `--data-root`/`--parent-root` 优先于专属环境变量。
旧 neg11/repair 数据目录在任何日志、锁或验证记录写入前即被保护。
下面三个命令全部在 CPU 节点运行；前一个成功后再运行下一个：

```bash
# 1. 只读历史 raw、gate、merged、GT、原评分结果；不加载模型
bash shell/ui14_alignment_context_a800.sh audit-errors \
  --task ui_alignment --steps 2000 4000 &&

# 2. 冻结已完成 neg11 available 产物，核对实际训练标签，生成独立清单和正式 YAML
bash shell/ui14_alignment_context_a800.sh prepare &&

# 3. 在 CPU 上复制旧 2000/4000 的模型文件，边复制边 SHA256；准备补评身份
bash shell/ui14_alignment_context_a800.sh eval-prepare \
  --task ui_alignment --steps 2000 4000
```

### 旧环境变量导致串目录后的接续

若 533f36b 的审计曾落在旧 neg11 目录，且 prepare 因父目录缺少
`negative_extension_manifest.json` 失败，更新代码后使用以下命令。
已完成的审计只读复制到新目录，再核对原输入及代码身份；匹配时日志显示 reused，
不重新评分、不推理。旧目录中的数据和审计文件不移动、不覆盖、不删除。

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui14-alignment-context
git pull --ff-only

export UI14_ALIGNMENT_DATA_ROOT=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_data/ui14_alignment_context_v1
export UI14_ALIGNMENT_PARENT_DATA_ROOT=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_data/ui14_cpt9000_neg11_v1

bash shell/ui14_alignment_context_a800.sh audit-errors \
  --task ui_alignment --steps 2000 4000 \
  --reuse-audit-root /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_data/ui14_cpt9000_neg11_v1/historical_audit &&
bash shell/ui14_alignment_context_a800.sh prepare &&
bash shell/ui14_alignment_context_a800.sh eval-prepare
```

`historical_audit/import_summary.json` 记录 imported/reused/skipped 数量。
缺失或校验不通过的审计不会冒充完成，原 audit-errors 会补做需要的 CPU 审计。
导入不改变原审计身份算法，也不改变生成、评分、数据选择、训练参数或正式 YAML。

如果旧 run/对应 test 不在默认路径，给 `audit-errors` 和 `eval-prepare` **同时**加上
`--old-run /实际旧run --old-manifest /对应旧数据/evaluation_manifest.json`。
入口核对该 step 已完成状态中的 manifest 摘要，不允许套用新的 neg11 test。
审计不需要旧权重仍然存在；补评要求原 checkpoint 或本次已准备的独立模型快照可用。
缺少文件会明确失败，不生成虚构的历史数量。

随后独立申请评测 GPU；不占当前训练使用的四张卡：

```bash
# 只补评 ui_alignment，默认一张独立 GPU。两个 step 依次完成。
# 不复制权重、不执行 detector；只消费 CPU eval-prepare 的结果。
bash shell/ui14_alignment_context_a800.sh eval-existing \
  --task ui_alignment --steps 2000 4000 --gpu-devices 0
```

完成后释放这份评测资源，在 CPU/提交节点检查并提交新的正式四卡 A800 训练：

```bash
bash shell/ui14_alignment_context_a800.sh check &&
bash shell/ui14_alignment_context_a800.sh submit --resource-group aiai_locate
```

使用现有 mlx 提交器；group_id=2146，
queue=`compute-3302-yg-cloudnative-ai-aiai.locate-guarantee`，
资源为 `ies_aiai_experience/AIAI_locate`、单节点四张 A800 40GB。
只渲染提交产物可用 `bash shell/ui14_alignment_context_a800.sh render`。
状态可用 `bash shell/ui14_alignment_context_a800.sh status`。
每阶段保留 `progress.json` 和 `stage_summaries/<stage>.json`，默认每 10 秒打印进度。

## worker 导入 eaglevl.ui_task_registry 失败后的接续

若 worker 在模型加载前报 `ModuleNotFoundError: No module named 'eaglevl.ui_task_registry'`，
更新本分支。推理脚本现在先加入自身 checkout 根路径，再导入 eaglevl 和模型依赖，
避免 conda 环境里另一个 editable checkout 抢先被导入；日志打印 `[inference imports]`
及实际包路径。若本 checkout 的文件确实缺失或旧包已被启动钩子提前加载，会在模型加载前
报告具体路径，不会静默切换到另一版本。无需重装 torch/CUDA 或修改评测并发数。

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui14-alignment-context
git pull --ff-only &&
bash shell/ui14_alignment_context_a800.sh eval-prepare --task ui_alignment --steps 2000 4000

# 上一步成功后，在独立评测 GPU 上运行
bash shell/ui14_alignment_context_a800.sh eval-existing \
  --task ui_alignment --steps 2000 4000 --gpu-devices 0
```

本次推理文件摘要改变，eval-prepare 刷新比较身份与目录；已完成且来源/属性匹配的
checkpoint 快照文件仍然复用。原 raw 审计、冻结数据和 detector 缓存无需重做。
这次导入失败的 worker 尚未开始模型推理，原失败目录保留。若之前使用自定义
`--old-run`/`--old-manifest`，eval-prepare 继续传入相同参数。
本次 CPU 验证见 [导入修复报告](ui14_alignment_import_cpu_report.md)。

## 成功标志和产物

| 阶段 | 成功标志/输出（相对新数据目录） |
| --- | --- |
| audit-errors | `historical_audit/latest.json` status=complete；`summary.csv` |
| audit-errors 逐 step | `historical_audit/ui_alignment/step-N/<audit_id>/summary.json`、`images.jsonl`、`missing_raw.jsonl`、`samples.html` |
| prepare | `cpu_check_report.json` ready=true；`frozen_parent.json`、`alignment_answer_contract.json` |
| prepare 清单 | `training_recipe.json`、`task_registry.json`、`evaluation_manifest.json`、`formal_job.yaml`、`formal_runtime.json` |
| eval-prepare | `comparison_plan.json` complete=true；`model_snapshots/checkpoint-N/snapshot.json` complete=true |
| eval-existing | `comparisons/step-N/<comparison_id>/complete.json` status=success，独立 predictions、raw、gate、merged 和 metrics.json |
| submit | 现有提交器返回 mlx 任务信息；正式提交 YAML/绑定记录在新目录 `submissions/` |

HTML 默认每种错误取固定 5 个样例，`--examples 10` 可调整。样例图片内嵌、红框为预测，
GT 在关闭的 details 中，点击显示；没有原始答案的记录明确显示缺失。
审计同时保存语法分类和原评分器的 invalid 标志，两者不能混为一类：
旧解析器可能接受包含坏文本的答案中的合法框。错误主分类每图仅一个；
非法输出的正/负原图数及 image/bbox TP/FP/FN/TN 贡献单独统计，bbox 不定义 TN。
逐图贡献求和必须等于原 merged 的评分及已保存指标；不相等则输出 mismatch 并返回失败。
历史 2000/4000 核对值只作为提示列，不能当作此次已读取的实测值。

## 数据、生成与恢复契约

prepare 不运行 inventory、normalize、detector 或裁图，不重新搜索负图。它复用已冻结的
available selection、normalization、eval_set 与真实正负数；七项 synth 正负俱全时仍按 1:1
训练采样，全局 negative:positive 仍为 2。小型可变清单在新目录独立保存；
清单中的原始图、派生图、已完成标签和 detector 几何可以指向旧不可变文件。
没有给旧 JSON、done 标记或报告建立可写硬链接。父摘要或输入变化会拒绝混入本目录。
旧 repair-v2 Excel 是历史定位材料，不混入 neg11 新 test 的指标历史。

新增生成策略为 `UI_EVAL_ANSWER_GRAMMAR=ui14_answer_v1`，只在新 profile 和独立补评启用；
旧 profile 默认 legacy。它逐 token 校验完整答案前缀：

- 正例沿用 `<ref>任务 prompt_label</ref><box><x1><y1><x2><y2></box>`，可以有多个框。
- 负例沿用训练标签 `<box>none</box>`，结束后只允许 EOS，不再接坐标。
- 标签文字、框间连接和终止使用 AR；正例 box anchor 后保留原六位置 MTP/PBD。
  MTP 回答与 AR 接续共用同一前缀。未接受的 MTP 帧回退时恢复 KV 和 slot/PBD 状态；
  不把非法输出后处理成 none。
- token 预算内未完成结构仍为失败；IoU=0.1、原解析/评分惩罚及分母不变。

新推理 manifest/raw、完整周期评测身份与补评绑定都包含解码策略/代码摘要。
改动解码代码后需重新运行 CPU eval-prepare，生成新的补评目录，然后重新推理。
原 raw 审计继续按来源摘要复用。权重通过 CPU 快照保持字节相同，推理兼容文件即使需要补齐，
也只写独立副本；旧 checkpoint config、Excel 和状态文件不写入。
同一入口重复运行可恢复已完成审计、逐文件权重副本和单图推理；完整补评直接复用。

新正式 run 从同一 CPT checkpoint-9000 初始化，新 SFT optimizer/scheduler 从 0 开始。
仍为 16k SFT、SDPA/BF16、ZeRO-2 two-lr、batch=1、GA=2、7268/12800 token 限制、
seed=42、warmup=500、cosine、LR=1e-5/2e-5、weight_decay=.01、max_grad_norm=1。
每 100 optimizer steps 写训练 Excel；step 0 及每 1000 steps 完整评测 14 项/36 行，
console 保留 UI5 五类摘要，Excel 保留两个 sheet；UI5 best 字段与每 4k 正式 checkpoint
规则不变。每 GPU 两个评测槽位、synth_loneword 独占物理卡、EVAL_FAIL_POLICY=stop 均继承。
新 Excel 为新训练输出下的 `diagnostics/ui5_training_evaluation.xlsx`。

参考 YAML 在 [ui14_alignment_context_a800.yaml](ui14_alignment_context_a800.yaml)；
它已由正式渲染器及参数校验器生成，实际提交仍须先完成本批数据的 prepare/check。
CPU 验证与实际执行范围见 [CPU 报告](ui14_alignment_context_cpu_report.md)。

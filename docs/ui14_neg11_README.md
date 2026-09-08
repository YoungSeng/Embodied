# UI14：七个合成任务补充 1:1 正常原图

基于 `e06add6b0b3c05b3f66384cf93331a0c94c076e8`，分支 `codex/m32-cpt9000-ui14-neg11-v1`。旧训练继续运行。所有命令从新 checkout 执行；入口拒绝新输出与旧数据、旧代码、原始 UI9 或旧模型输出目录重叠。

## 独立 checkout 与默认路径

在开发机执行以下命令，只创建新目录，不在旧 checkout 内操作 Git：

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4
git clone --single-branch --branch codex/m32-cpt9000-ui14-neg11-v1 \
  https://github.com/YoungSeng/Embodied.git Embodied-ui14-neg11
cd Embodied-ui14-neg11

export UI9_DATA_ROOT=/mnt/bn/intelligent-service-yg/dataset/gui/ui9_datasets_v1
export UI14_PARENT_DATA_ROOT=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_data/ui14_cpt9000_repair_v2
export UI14_DATA_ROOT=/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_data/ui14_cpt9000_neg11_v1
```

显式设置这三个变量，避免终端继承上次实验的 `UI14_DATA_ROOT`。默认 Python 为 `/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/conda_envs/LocateAnything/bin/python`；可用 `UI14_PYTHON` 指定 CPU 环境。OCR 继续使用旧 detector 配置记录的 UI5PaddleOCR 环境。

| 项目 | 路径（共同前缀 `/mnt/bn/intelligent-service-yg/logging/sicheng_workspace`） |
|---|---|
| 旧代码，只读 | `code/Eagle_LocateUI5_v4/Embodied-ui14-cpt9000` |
| 新代码 | `code/Eagle_LocateUI5_v4/Embodied-ui14-neg11` |
| 旧数据，只读 | `gui_data/ui14_cpt9000_repair_v2` |
| 新数据 | `gui_data/ui14_cpt9000_neg11_v1` |
| 旧训练，只读 | `gui_models/locany-m32-cpt9000-ui14-a800x4-repair-v2` |
| 新训练 | `gui_models/locany-m32-cpt9000-ui14-neg11-a800x4-v1` |
| 初始化，只读 | `gui_models/locany-3b-ui-cpt-v4-v3-h20x2-formal-segmented-eval/checkpoint-9000` |

## 按资源执行

所有 CPU 阶段默认 16 workers，单协调进程加锁、有界图片队列。阶段失败后执行原命令恢复。不要同时启动两个写入同一新数据目录的准备命令。下面各阶段都只写新目录。

### ① CPU inventory

```bash
bash shell/ui14_neg11_a800.sh inventory
```

读取旧规范化快照、原 train/test、source_metadata、复制 SQLite/JSON 清单及各来源 `sample_imgs/_folders/`。输出 `inventory_summary.json` 配额表、`negative_candidates.jsonl`、`negative_rejections.jsonl`、`negative_selection.proposed.jsonl`、`negative_page_assignments.json`、`negative_split_overlap.json`、`normal_pairs.html`。

成功标志：`stage_summaries/inventory.json.status=complete`。配额表的 `gap=0` 才能进入 normalize；inventory 即使有缺口也保存完整盘点结果。目标由旧规范化数据的独立 RGB 内容身份计算，用户给出的 9,600 张正例 test 只作参考核对。

配对 `LocalImgURL` 采用导出器的正常图语义；`RawImgURL` 还需明确的合成前底图或当前任务 clean 证据。路径命中、其他任务负标签、无标注、名为 raw 的目录均不构成 clean 证据。缺口不会通过复制行填满。

若现有引用不足，可提供来源已验证的正常池 JSONL，写在新数据目录的 `normal_pools/<task_key>/*.jsonl`，再运行 inventory。原始来源里已经存在的 `normal_pool*.jsonl` / `negative_pool*.jsonl` 也只读检索；不修改原始 UI9。备份、quarantine、.work 中的池不纳入。每条至少：

```json
{"image":"sample_imgs/_folders/version/local_imgs/page.png","source_dataset":"来源名称","source_version":"版本","FigmaKey":"设计文件","FigmaNodeID":"页面节点","negative_evidence":{"clean_tasks":["synth_radius"],"basis":"已有人工核验结果或数据生成记录","provenance":"可追溯的记录 ID/文件"}}
```

同源池优先，跨源补足单独记录数量和占比。已知 Figma 页面、raw 父图引用及内容身份继承冻结 split；冲突候选拒绝。未知组用 seed=42 的确定性映射，七项共享保存结果。不根据模型预测选择。页面字段缺失数量单独报告，不猜测不存在的页面 ID。

HTML 每任务最多 10 对正常/异常图，GT 默认隐藏；通过能访问挂载盘的本机文件浏览器打开。它引用原图，不复制大图。池候选没有配对异常图时仅显示正常图和证据。

### ② CPU normalize

```bash
bash shell/ui14_neg11_a800.sh normalize
```

冻结 `negative_selection.jsonl` 和 `negative_extension_manifest.json`。正常记录的主图是正常路径，`boxes_px=[]`、`is_positive=false`、合法 `<box>none</box>`。旧正例坐标和 split 保持原值，不重复做 375 投影；只给组合数据建立自己的 normalization ID。

输出 `normalized/<task>/train.jsonl`、`test.jsonl` 和 detector inputs（准确路径以 registry/manifest 为准），`source_snapshot.json` 保留父修复信息，`normalization_stats.json` 分列 reused/new。成功标志为 `complete=true` 和 CPU 报告 `normalization_complete=true`；此时 ready 仍为 false。

快照已经冻结后不能原地换一批正常图；若确需换数据，使用另一个独立数据目录。中断后可继续 normalize，只补缺失产物；已完成再次运行直接复用。

### ③ CPU cache-prepare

```bash
UI14_PREPARE_WORKERS=16 bash shell/ui14_neg11_a800.sh cache-prepare
```

私有复制旧 JSON、分片、done 标记、几何索引和验证日志。PNG 保持指向旧只读路径；没有可写硬链接。稳定分片成员和顺序保留，新图追加分片。扫描/内容 hash 仅检查首次出现或属性改变的图片。

六个合成 crop 任务新增正常图；`change_line_illegal_v3` 的两个旧 split 也保留独立交接清单，合计仍为七个 crop 任务 × train/test。新增大间距任务正常图保持全图。

按完整 detector 文件字节 BLAKE2b ID、尺寸及配置查找已完成跨任务检测；已命中部分放入分片 seed，GPU worker 只处理缺项。RGB SHA256 只作规范化身份，不作为 detector ID。异常图与正常图内容不同就不会命中。跨任务索引使用本次 CPU 准备时已完成的检测；以后新完成的跨任务结果可通过再次 cache-prepare 纳入。

成功标志：全部 14 个 `cache/<task>/<split>/manifest/ui14_prepare_ready.json` 通过绑定校验；查看 `cache_preparation/summary.json`、`cross_task_detector_reuse.json` 和 `cache_reuse_summary.json`。完成后才申请另一份四卡 GPU。

### ④ 独立四卡 A800 cache

```bash
bash shell/ui14_neg11_a800.sh cache
```

只执行缺失 OCR/OmniParser，消费 CPU 交接 JSON；不会退回图片扫描或 CPU 裁图。旧训练的四张卡继续训练，本步骤必须使用另一份已分配或排队资源。

成功标志：`stage_summaries/cache.json.status=complete`，`cache_reuse_summary.json` 所有 text/icon 的 pending=0。分开记录 parent_reused、cross_task_reused、new_inferred_completed。完整分片可续跑；中断的未完整分片按现有生命周期重跑其中未有可靠检测记录的图。

检测结束立即退出，释放该份调试 GPU。

### ⑤ CPU cache-finalize

```bash
UI14_CROP_WORKERS=16 UI14_PNG_COMPRESS_LEVEL=1 \
  bash shell/ui14_neg11_a800.sh cache-finalize
```

合并增量检测，复用原图对应的 GT-free 几何、旧 PNG 和物理完成记录；标签绑定新的 normalization ID。只裁新增/变化图片，标签变化不会导致 detector 重算。仍按唯一原图一次解码、低压缩无损 PNG、原子发布及逐图完成记录续跑。

成功标志：14 个 crop split 完成 `ui14_crop_complete.json` / `ui14_label_cache_ready.json`，新目录 `derived/` 与覆盖统计齐全。计时和吞吐延续现有 `crop_performance/` 报告；复用数不计入新建吞吐。

### ⑥ CPU finalize 与 check

```bash
bash shell/ui14_neg11_a800.sh finalize &&
bash shell/ui14_neg11_a800.sh check
```

直接组装已完成标签和缓存，不重新解码/裁图。输出：

- `training_recipe.json`、`task_registry.json`、`evaluation_manifest.json`：14 项。
- `negative_image_counts.json`：每 task/split 独立原图正负 1:1、派生 crop 正负数。
- `sampling_stats.json`：共享 sampler 的真实抽样模拟，含 clean_source_image/background_crop 数量。
- `cpu_check_report.json`：父/子身份、路径、标签、坐标、分片绑定、重叠及检查结果；成功为 ready=true。
- `formal_job.yaml`、`formal_runtime.json`：新目录四卡正式配置。

仅七个 synth recipe 声明 `negative_to_positive_ratio=1.0`，已接到真实训练 sampler；全局 `UI_NEGATIVE_TO_POSITIVE_RATIO=2` 和其余七项旧 recipe 不变。任务均衡、源图均衡、crop 轮换不变；正常裁片继续作为 background_crop，完整正常图作为 clean_source_image。

需要独立全量 CPU 内容复核时：

```bash
bash shell/ui14_neg11_a800.sh check --full-verify
```

此入口会读取图片内容，放在 CPU 节点运行。通常 check/submit 复用有来源和文件属性依据的验证记录。

### ⑦ 新的四卡正式训练

```bash
bash shell/ui14_neg11_a800.sh submit
```

入口固定传递 `--profile m32-cpt9000-ui14-neg11-v1 --machine a800 --resource-group aiai_locate --gpus 4`；可显式追加 `--resource-group yg` 使用现有另一个资源配置。默认资源是 ies_aiai_experience/AIAI_locate、group_id=2146、queue=compute-3302-yg-cloudnative-ai-aiai.locate-guarantee。

新实验从 CPT-9000 开始新的 optimizer/scheduler、SFT step=0，16,000 steps、每 100 步训练 Excel、step 0 及每 1,000 步全量 14 项评测、每 4,000 步正式 checkpoint，原 UI5 best 字段语义不变。SDPA/BF16、ZeRO-2 two-lr、7268/12800 token 限制等参数沿用原 profile。`EVAL_FAIL_POLICY=stop`、observe gate、每 GPU 两槽及 synth_loneword 独占一张物理卡不变。

提交时继续执行原入口的快速检查、进度与提交状态逻辑。任务 ID 以 mlx 成功响应为准；单纯生成 YAML 不是提交成功。只预览提交可用 `submit --render-only`。

## 状态、评测身份和旧模型比较

```bash
bash shell/ui14_neg11_a800.sh status

# 新数据 ready 后，在独立 GPU 资源运行
bash shell/ui14_neg11_a800.sh eval-existing \
  --checkpoint /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_models/locany-m32-cpt9000-ui14-a800x4-repair-v2/checkpoint-1000

# 可选 CPU：只读重汇总旧 step 1000 预测和错误类型
bash shell/ui14_neg11_a800.sh audit-errors
```

每 10 秒打印阶段、task/split、数量、耗时、当前阶段 ETA、复用/新增信息和输出目录；无法估算的阶段明确显示估算中。实时数据在 `progress.json`、`progress/*.json`，阶段结果在 `stage_summaries/*.json`。SIGKILL 后可能保留 running；结合该文件的 pid/更新时间判断，重新执行同一阶段恢复。

新 Excel 在新训练输出的 `diagnostics/ui5_training_evaluation.xlsx`，保留 train_100steps/eval_1000steps。每轮 14 项 image/bbox 加 UI5、UI9 各自 macro/micro，共 36 行；UI5 真实 TP/FP/FN/TN 完整写入，micro 按计数计算，缺失保留为空，bbox 不定义 TN。PBD/coordinate bridge/slot routing 从 tile 汇总，未知/不一致不冒充 False。

新 test 的 eval_set_id、数据摘要和独立正负数随结果保存。旧正例子集从同轮预测另外评分到 `evaluation/raw/<独立 attempt>/parent_positive_metrics.json`，无需二次推理，不把新 test F1 与旧 test F1 混成历史增减。

eval-existing 验证旧 checkpoint 的 14 个任务 ID/结构/输入策略，用显式新评测集覆盖数据绑定，禁止 patch 旧 config；输出到新数据目录 `comparisons/<真实权重及 processor 摘要>/<eval_set_id>/`。没有可靠身份匹配的旧预测不导入；相同独立比较可复用自身逐图结果。普通训练仍严格验证 checkpoint 数据绑定。

inventory 会在旧 step 1000 预测可读时自动生成 `parent_parse_error_audit.json`；audit-errors 还在旧评测状态完整时用 CPU 重新评分 UI5/UI9，写入新 `parent_prediction_audit/<独立 attempt>/`。区分空输出、框解析、格式非法和运行失败，并附少量原答案示例。原输出、评分惩罚、分母和运行中的旧 Excel 都不改写。

## 本次交付的执行边界

见 [CPU 验证报告](ui14_neg11_cpu_check_report.md)。本地仅执行 CPU 构造数据测试、代码/路径隔离检查和正式 YAML 渲染，没有连接 A800；真实候选/缺口、生产缓存复用数量、GPU 吞吐和正式任务 ID 尚无实测值。参考 YAML 随代码保存在 [jobs/rendered/ui14_neg11_a800x4.yaml](../jobs/rendered/ui14_neg11_a800x4.yaml)，最终运行使用 finalize 生成并绑定真实数据的 YAML。

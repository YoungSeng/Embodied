# UI14 step-0 启动恢复 CPU 检查（2026-09-09）

基于 neg11 分支 `c1efd6be227953b4451e4a26495ef30ddf82a398`。
用户提供的正式日志已完成 step 0 的 14 项评测和 36 行 Excel，在 0→1000 训练前的数据绑定
校验失败。本地只执行代码修改及 CPU 回归，没有连接 A800、读取该 run 文件或重新提交集群任务。

`21079cc` 首版恢复校验还有一个兼容错误：只认可导出器载入前临时设置的
`checkpoint-0-export`。真实 `initialize_or_validate_ui_relation` 随后会调用模型初始化，
将 config 中的 reason 改成 `all-ui-relation-keys-missing-checkpoint-0-export`，并把相同
seed/reason 保存到 config 初始化统计和 `ui5_checkpoint0_manifest.json.initialization`。
本次更正校验，接受真实最终值并交叉检查报告；没有修改原 checkpoint 或导出/初始化实现。
上一轮手写成功夹具没有覆盖 reason 被覆盖这一行为，本轮改用生产代码元数据调用链生成夹具。

## 根因与修改

- `scripts/ui14_profile.py`：保持非零 SFT 断点错批保护，兼容完整、来源与任务表匹配且无
  optimizer/trainer/RNG 状态的旧 checkpoint-0；只读提交检查不写绑定，训练启动时原子写入。
- `shell/run_locany_ui5_pipeline.sh`：在 checkpoint-0 导出/评测之前调用既有 CPU ready 校验与绑定。
  eval-only 分支保持原行为。UI14 完成检查改由恢复入口调用；UI5 完成检查保持原逻辑。
- `scripts/ui14_run_recovery.py`：跨本次启动修复核对既有 step-0 评测，保存独立复用凭证。
  核对完整任务/Excel 集合、manifest/eval_set、模型结构、运行配置、CPT 来源、批次、Git 变更，
  并要求 checkpoint 文件时间早于评测开始。模型权重不加载，不重读大文件计算内容摘要。
- `tests/test_ui14_run_recovery.py`、`tests/test_ui14_initial_evaluation.py`：新增故障复现，执行真实
  Bash 编排片段，使用 CPU 文件与 Excel 夹具替代 GPU 推理。

## 已实际执行

本轮 22 项 CPU 测试全部通过（9.800 秒），另通过两个变更 Python 文件 AST 检查及实际 Git
变更兼容检查。Bash 入口本轮未改；此前两个入口语法检查通过。

- 首次运行先发布绑定，重复启动不改写绑定；错批被拒绝。
- 旧无绑定 checkpoint-0 只读核对后可恢复；核对不读取权重内容或修改 config。
- 成功夹具执行生产导出器、加载状态分支、模型初始化元数据赋值和 manifest 序列化的 CPU
  语句，确认最终 reason 覆盖临时值，并核对两份初始化报告一致。模型/张量加载与参数统计
  使用替代对象，没有初始化真实模型、执行 tensor 初始化或使用 GPU。
- 实际最终 reason 和旧短 reason 都能恢复；错误/缺失的最终初始化报告、报告 seed/reason
  与 config 冲突、训练初始化 reason 均被拒绝。CPT 路径、任务表、训练状态等检查照旧。
- 错误 CPT/seed/任务表、缺导出完成标记、混入训练状态均被拒绝。
- 即使 checkpoint-0 完整，存在无绑定 checkpoint-1000 仍被拒绝；原断点完整性检查继续通过。
- 相同数据/配置下，启动代码更新后可两次复用原 14 项/36 行结果；模型、评测 JSON、Excel
  的字节与修改时间完全不变，只新增独立复用凭证。
- UI5-only/失败评测、缺 Excel、数据或运行配置变更、权重时间变化、推理/评分代码变更、
  未提交的推理代码变更、缺失历史 Git 提交都不会获得兼容复用。编排脚本仅允许本次精确的
  绑定插入与完成检查入口替换；直接修改 Bash 中的 generation 参数也会拒绝复用。
- 实际 Bash 片段验证绑定→导出→评测→训练顺序；完整评测跳过导出和推理；绑定失败或
  评测失败停止训练。原推理单图重试、checkpoint-0 非训练断点、36 行及 UI5 汇总语义回归通过。

复现测试：

```bash
python -m unittest \
  tests.test_ui14_run_recovery \
  tests.test_ui14_initial_evaluation \
  tests.test_ui14_inference_resume \
  tests.test_ui14_repair.RepairedIntakeTests.test_sft_resume_cannot_cross_repair_batches \
  tests.test_ui5_pipeline.CheckpointTests.test_checkpoint_zero_is_not_a_training_resume_candidate \
  tests.test_ui5_pipeline.CheckpointTests.test_training_candidates_cli_excludes_checkpoint_zero \
  tests.test_ui5_pipeline.CheckpointTests.test_latest_resume_never_falls_back_past_newer_incomplete_checkpoint \
  tests.test_ui5_pipeline.CheckpointTests.test_sharded_model_requires_every_indexed_shard \
  tests.test_ui14_pipeline.UI14EvaluationTests.test_excel_requires_36_rows_and_keeps_ui5_macro_independent -v
```

本次不生成新数据版本、recipe 或 YAML 配置变体；继续使用既有四卡 neg11 profile。
真实运行是否复用及恢复训练，以更新后启动日志和 `evaluation/ui14-step-0-startup-reuse.json` 为准。

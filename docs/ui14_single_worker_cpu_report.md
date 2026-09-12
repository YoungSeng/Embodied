# UI14 全任务单卡单进程评测：CPU 验证

日期：2026-09-12。基线：`b0a0e74faabcf359940f3dc3f4fb3b9e0d85b118`。
分支：`codex/ui14-neg11-alignment-crops-v1`。

用户提供的 step-3000 日志确认 `synth_radius` 在同卡两个进程占用显存时 OOM。
按最新要求，全部 14 个任务统一每物理 GPU 一个推理进程，四卡最多同时运行四个任务。
旧环境、旧 YAML 和显式命令中的每卡 2 个参数兼容接收并降为 1；独立旧 checkpoint 补评也使用 1。
UI14 runtime、评测状态和调度器将全部 14 项记为独占，不依赖旧独占任务子集。

## 改动

- `configs/ui14_cpt9000_formal.json`、`locany_ui5_common.py`、`ui14_checks.py`：正式值、旧环境兼容和检查一致。
- `run_ui5_eval.py`、`run_ui14_eval.py`、`ui14_alignment_eval.py`：周期/独立评测均只传一个推理进程。
- `run_ui5_parallel_inference.py`：底层也限制每卡一个进程，按预计耗时领取任务；原子保存 running/pending/失败状态，末尾再次打印失败任务与原因。
- `ui14_incremental_eval.py`：每 10 秒区分成功/失败数，显示正在运行和未启动任务；失败后的等待明确说明；最终异常保留 task/GPU/原因/日志路径。
- README、正式参考 YAML/runtime 及 CPU 回归随代码更新。

保留逐 task 的 image/bbox 两行原子写入、完整 14 项/36 行后才提交 macro/micro/best/history、
`EVAL_FAIL_POLICY=stop`、原始预测身份及复用、原评分惩罚。无自动 OOM 重试或生成规则变更。
数据、模型结构、训练 batch/梯度累计/学习率、CPT9000 初始化、16k SFT、100/1000 步记录均未改变。

## 实际执行

20 项不同 CPU 回归通过：16 项核心测试，另 4 项评测/评分集成测试。
首轮有两个旧测试断言/依赖问题，已修正；两个真实评分子进程还遇到 Windows GBK 控制台无法输出 emoji，
使用 UTF-8 测试环境复核通过，没有为此修改生产评分器。

- 真实 CPU 子进程验证四卡模型槽位最多四个，同一卡全部任务的进程区间不重叠；14 项各执行一次。
- 显式旧值 `--workers-per-gpu 2`、旧独占任务列表、正式环境值 2 均执行每卡 1 个。
- 子进程失败仍停止领取新任务；已运行任务的成功结果继续保存，无伪造成功。
- 失败任务/当前 task/GPU/未启动清单持续可见，失败事件不计入成功；读取追加日志时不混入上次 OOM。
- 降低并发不改变预测身份，已完成图片和 checkpoint 字节保留；重跑只推理 OOM/缺失图片。
- 逐任务 Excel 在其他进程失败前可见；中断后成功任务行保留，续跑补齐 14 项/36 行，训练 sheet 和旧 attempt 保留。
- step 0、独立旧 checkpoint 的正式 legacy 解码、真实 UI5/UI9 评分和历史身份隔离继续通过。
- 使用正式 renderer 生成并校验 `docs/ui14_neg11_alignment_crops_a800.yaml` / `ui14_neg11_alignment_crops_runtime.json`：
  4×A800、每卡 1 个、CPT9000、16k SFT、100/1000 步、stop，输出目录保持当前 run。

本地日志：`work_dirs/single_worker_core.log`、`work_dirs/single_worker_integration.log`、
`work_dirs/single_worker_integration_recheck.log`（不提交测试产物）。

## CPU 复核命令

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python -m unittest -v \
  tests.test_ui14_inference_workers \
  tests.test_ui14_incremental_eval \
  tests.test_ui14_inference_resume \
  tests.test_ui14_alignment_crops.CropAlignmentTests.test_formal_yaml_is_four_a800_cpt9000_with_production_decoder_and_exclusive_tasks \
  tests.test_ui14_alignment_crops.CropAlignmentTests.test_all_tasks_reserve_physical_cards_and_four_gpus_stay_parallel \
  tests.test_ui14_alignment_crops.CropAlignmentTests.test_independent_comparison_selects_legacy_even_with_stale_grammar_env \
  tests.test_ui14_pipeline.UI14EvaluationTests.test_partial_oom_results_survive_and_resume_commits_full_36_rows \
  tests.test_ui14_pipeline.UI14EvaluationTests.test_step_zero_evaluation_writes_all_14_tasks_and_resumes_without_fake_training \
  tests.test_ui14_pipeline.UI14EvaluationTests.test_real_scorer_retries_preserve_step_1000_reports_and_predictions
```

## 集群续跑

本地环境没有访问 A800；未执行真实 step-3000 GPU 补评或提交集群任务，没有新的集群任务 ID。
用户日志中的 10/14 个已评分任务已落 Excel，单张图片是否可复用以原有身份检查为准。

确认失败作业及其 worker 已退出后，在当前 checkout 执行：

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui14-neg11-alignment-crops
git pull --ff-only origin codex/ui14-neg11-alignment-crops-v1 &&
bash shell/ui14_neg11_alignment_crops_a800.sh submit --resource-group aiai_locate
```

保持当前输出目录；不删除 checkpoint/预测/Excel，不传 `--overwrite`，无需重做数据或缓存。
入口校验最新完整 checkpoint-3000 的 optimizer/scheduler/rank 状态，补齐本步评测后继续训练。
这能消除本调度器内部的共卡模型显存竞争；实际 A800 峰值和补评结果仍需集群运行确认。

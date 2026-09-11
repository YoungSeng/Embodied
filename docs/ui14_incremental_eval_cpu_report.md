# UI14 评测 OOM 隔离与逐任务 Excel：CPU 验证

日期：2026-09-11。基线：`d79e00c38eae6efaab88b6c636c9c9bfffe32caa`，
分支：`codex/ui14-neg11-alignment-crops-v1`。

本次仅修改评测资源调度、完成事件、评分落盘和相关配置/测试。
推理生成文件、模型权重、prompt、crop 几何、数据清单和训练参数没有改动。

## 实际执行结果

35 项不同的 CPU 回归通过（32 项套件中的 30 项通过；修正两个模拟评分夹具后
2 项复核通过，另有 3 项初始评测/Excel 兼容性通过）。
评分使用现有 UI5/UI9 评分器，CPU 子进程模拟任务并发与失败，不加载模型或 CUDA。

- 四卡双槽调度：synth_loneword、change_line_illegal_v3、synth_inner_margin
  三项分别占满各自物理卡的槽位；其他任务并行并在卡释放后继续领取。
- 老环境只配置前两项时仍加入 synth_inner_margin；正式 YAML 已用实际 renderer
  生成并通过 validate_formal_yaml。维持 CPT9000、16k steps、100/1000 记录、
  四卡 A800、legacy/hybrid、现有可用负样本规则、stop 失败策略。
- 真实 CPU 子进程先发完成事件，等待 Excel 已发布后再模拟另一任务失败，
  最后发出第三个成功事件；两个成功任务均已写入，失败任务没有伪造指标。
- 完整评测夹具在第六项写完后模拟 OOM：保留 6 项/12 行，整轮标为失败，
  best/history 不提前提交；续跑最终为 14 项/36 行且同一步不重复。
- 14 项全齐前只有任务行，不写不完整集合的 macro/micro。训练 sheet 保留；
  不同 eval_set/data 摘要拒绝混写；已有完整同身份结果保留。
- 实际 TP/FP/FN/TN、无负样本来源、非法输出惩罚、UI5 micro 和 tile gate 状态
  沿用现有评分；bbox TN 保持空值。view_policy 从本轮任务结果读取。
- 旧预测复用测试验证降低并发不改变推理身份，仅重试 OOM/缺失图片，
  已完成预测与 checkpoint 文件字节不变。评分 attempt 独立保存。
- step-0 初始评测、成功跳过、失败停止与旧 Excel 表头迁移回归通过。

## 复核命令

从本分支根目录，在已有 CPU 依赖的 Python 环境执行：

```bash
python -m unittest -v \
  tests.test_ui14_incremental_eval \
  tests.test_ui14_pipeline.UI14EvaluationTests.test_full_evaluation_resume_repairs_missing_ui9_and_keeps_best_ui5 \
  tests.test_ui14_pipeline.UI14EvaluationTests.test_step_zero_evaluation_writes_all_14_tasks_and_resumes_without_fake_training \
  tests.test_ui14_pipeline.UI14EvaluationTests.test_real_scorer_retries_preserve_step_1000_reports_and_predictions \
  tests.test_ui14_pipeline.UI14EvaluationTests.test_partial_oom_results_survive_and_resume_commits_full_36_rows \
  tests.test_ui14_alignment_crops.CropAlignmentTests.test_three_oom_tasks_reserve_physical_cards_and_other_tasks_stay_parallel \
  tests.test_ui14_alignment_crops.CropAlignmentTests.test_formal_yaml_is_four_a800_cpt9000_with_production_decoder_and_exclusive_tasks \
  tests.test_ui14_inference_workers \
  tests.test_ui14_inference_resume \
  tests.test_ui14_neg11_eval \
  tests.test_ui5_excel_logger \
  tests.test_ui14_initial_evaluation
```

本地执行日志：`work_dirs/incremental_eval_validation.log` 和
`work_dirs/incremental_eval_recheck.log`（测试日志不纳入代码提交）。
当前环境未连接集群，未执行真实 checkpoint-1000 GPU 推理或正式提交。
独占卡消除本作业评测 worker 的同卡竞争，实际显存峰值需在 A800 复跑确认。
集群更新与续跑命令见 `ui14_neg11_alignment_crops_README.md`。

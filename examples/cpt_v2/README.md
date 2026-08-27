# CPT v2 schema example

本目录中的指标均为 synthetic/schema-only 示例，只用于验证 JSON/JSONL 字段、离线 Excel
重建和缺失值处理；不得用于模型结论、checkpoint 选择或采样推荐。

`diagnostics/` 模拟一个包含 VQA 与 UI defect 的极小 run。正式训练仍必须覆盖十个规范任务，
并以真实 `split_manifest.jsonl`、集群训练日志和 held-out evaluator 产物为准。

离线重建命令：

```bash
python scripts/build_locany_cpt_excel.py \
  --diagnostics-dir examples/cpt_v2/diagnostics \
  --output outputs/cpt_v2/cpt_training_evaluation.xlsx
```

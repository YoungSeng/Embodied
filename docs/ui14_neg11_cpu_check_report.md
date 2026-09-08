# UI14 neg11 CPU 验证报告

日期：2026-09-08。基线 `e06add6b0b3c05b3f66384cf93331a0c94c076e8`。实现位于独立 worktree / 分支 `codex/m32-cpt9000-ui14-neg11-v1`，原 checkout 仍在基线提交，原 tracked 工作文件没有修改。

## 实际执行结果

| 检查 | 实际结果 |
|---|---|
| 准备、采样、缓存、评测、提交、进度 CPU 回归集 | 155 tests，170.150 秒，OK |
| 最后新增正常池/raw 父图/独立 profile 修正后的定向回归 | 13 tests，41.191 秒，OK |
| 周期评测复用验证记录与评测兼容回归 | 25 tests，63.304 秒，OK |
| normalize 重建后的 ready 状态与全链路增量复用 | 1 test，27.349 秒，OK |
| 跨任务 detector 分片复用与原 worker 回归 | 9 tests，26.404 秒，OK（也包含在大回归集内） |
| Python AST | 22 个修改/新增 Python 文件通过 |
| Shell 语法 | `bash -n shell/ui14_neg11_a800.sh` 通过 |
| 正式配置 | 渲染并 CPU 校验独立 neg11 YAML/runtime；未提交 |
| 默认 A800 输入探测 | 失败退出码 1；Windows 主机没有 `/mnt/bn/...` 挂载，创建生产输出前退出 |
| A800 实际 inventory / normalize | 未执行 |
| A800 GPU detector / 模型推理 / 训练 | 未执行 |
| 生产缓存复用数、吞吐、剩余 ETA、任务 ID | 尚无实测值 |

这些测试用临时构造图片、JSONL、完成标记、缓存和 checkpoint 配置；模型推理及 detector 由 CPU 测试替身提供。实际运行了图片解码/裁图、坐标和标签转换、分片/绑定校验、真实 CPU 评分器、Excel 生成及 YAML 渲染。未加载训练模型，也未声称验证过 A800 显存或生产吞吐。

## 已验证的连接关系

- 任务仍为原 14 项；七个 synth 任务每个 train/test 的配额按独立正例内容计算。构造盘点选中 14 张 task/split 独立负例，重复引用不增加候选数；真实 9,600 张 test 目标留待生产旧规范化数据核对。
- 负例读取独立正常图片，主图与合成图不同，空 boxes、false is_positive、合法 none 答案；保留来源、配对 ID、页面和 seed。raw 目录名称/其他任务负标签不能通过证据检查。
- 内容路径别名和 raw 父图归属参与统一页面/split 约束；跨 split 冲突被拒绝并产生缺口，normalize 不会凑数。正常池清单可写入新数据目录，旧原始文件保持只读。
- 旧 repair/source_snapshot/normalization 的摘要和计数保留；扩展清单与自身 normalization ID 另行绑定。必须有全部 18 个规范化 split 及对应 detector inputs，失败计数必须为零。
- 旧分片成员/顺序不变，新图追加。旧 PNG 按原路径只读引用；旧 JSON/done/索引是私有副本，没有可写硬链接。构造测试比较旧文件字节和 mtime，确认没有改写。
- 只读复用旧正例图片、几何、PNG；正常图使用自身 detector 几何。标签重绑不触发旧图重新裁图；中断/缺项、单图/计划/标签变更的失效行为由原并行裁图回归覆盖。
- 跨任务 detector 复用按完整文件内容 ID、尺寸和配置匹配。两图构造分片中，一张从已有检测复用、一张实际调用假 detector；缓存图片的 Image.open 被禁止，验证没有解码它，输出顺序和原分片字节不变。再次 worker 不加载模型或图片，配置变化拒绝复用。
- 真实 sampler 读取每个 recipe 的 ratio，准备阶段调用同一采样计划；七项 1.0，其余七项保持旧条目和全局 2.0。正常原图负例与背景 crop 负例分别计数，在负池内保持源图均衡。
- 重复 normalize 保持已 finalize 的 recipe/registry；重复 finalize 在禁止 Image.open 的测试环境下通过。快速检查使用属性和内容证据，独立 full-verify 保留。
- 新评测完整 14 项/36 行；原 UI5 计数可还原 F1，micro 不将缺失 TP/TN 等填 0。bbox 没有 TN。tile 的 PBD/bridge/router 状态缺失或冲突保留未知。
- 外部旧 checkpoint 接受显式新 eval set，验证任务/结构且禁止 patch；旧 config 字节及 mtime 不变。14 项评测、独立 scoring attempt、缺项补齐、独立 eval_set_id、7 项原正例子集 CPU 评分、Excel 36 行已覆盖。
- 错误审计构造用例区分空答案、框解析、格式非法各 1 条，运行失败 1 条；原预测字节不变。用户报告的真实 465 条 parse_error 未在本机重新统计。

## 正式配置核对

参考产物：`jobs/rendered/ui14_neg11_a800x4.yaml` 和 `.runtime.json`。最终生产文件由 finalize 写入新的数据目录，并与实际 CPU 报告绑定。

单节点 A800 40GB ×4；ies_aiai_experience/AIAI_locate，cluster_id=24，group_id=2146，queue=compute-3302-yg-cloudnative-ai-aiai.locate-guarantee。独立 neg11 job name、checkout、data root、OUTPUT_DIR。

CPT checkpoint-9000 初始化、新 optimizer/scheduler、SFT step=0、16,000 steps；batch=1、accumulation=2、7268 sample/sequence tokens、12800 tokens/rank、BF16、SDPA、ZeRO-2 two-lr、LR=1e-5 / relation LR=2e-5、warmup=500、cosine、seed=42、weight_decay=.01、max_grad_norm=1。

每 100 steps 训练日志，step 0 + 每 1000 steps 全量 14 项评测，4k 正式 checkpoint；两槽/GPU、synth_loneword 独占一张物理卡、observe gate、IoU=.1、EVAL_FAIL_POLICY=stop。原五类 best / macro / micro 含义保持。

## 主要修改文件

| 链路 | 文件 |
|---|---|
| 单一入口、只读旧模型比较、错误审计 | `shell/ui14_neg11_a800.sh`、`scripts/ui14_neg11.py` |
| 正常图盘点、证据、配额、split、复合快照 | `scripts/ui14_neg11_data.py` |
| 私有缓存导入、跨任务 detector seed | `scripts/ui14_neg11_cache.py`、`scripts/run_ui5_crop_audit.py` |
| 复用几何、快速 finalize/check | `scripts/prepare_ui5_eval_detector_crops.py`、`scripts/ui14_neg11_finalize.py`、`scripts/ui14_repair.py` |
| 任务级 sampler 与训练负例类别统计 | `eaglevl/train/ui_defect_data.py`、`locany_finetune_magi_stream.py` |
| 计数、tile 状态、eval_set 与 Excel | `scripts/collect_ui5_metrics.py`、`locany_ui5_common.py`、`run_ui14_eval.py`、`eaglevl/train/ui5_excel_logger.py` |
| 独立 profile、YAML 与进度 | `scripts/ui14_profile.py`、`submit_locany_ui5.py`、`ui14_checks.py`、`ui14_progress.py`、`ui14_verification.py` |

CPU 回归可在已安装 Pillow、numpy/scipy、openpyxl、PyYAML 的环境运行：

```bash
python -m unittest \
  tests.test_ui14_neg11 tests.test_ui14_neg11_eval tests.test_ui14_neg11_detector_reuse \
  tests.test_ui14_detector_workers tests.test_ui14_inference_workers \
  tests.test_ui14_inference_resume tests.test_ui14_initial_evaluation \
  tests.test_ui14_pipeline tests.test_ui14_submission_resources tests.test_ui14_submit_progress \
  tests.test_ui5_excel_logger tests.test_ui5_pipeline tests.test_ui_sampler_cycle_cache \
  tests.test_ui14_cache_stages tests.test_ui14_parallel_crops tests.test_ui14_progress
```

最终新增正常池和评测验证记录用例后，这条命令包含 157 个用例；本次记录的是此前 155 项全量通过，随后 13 项数据/缓存/评测定向通过、25 项周期评测定向通过。运行资源与完整生产命令见 [README](ui14_neg11_README.md)。

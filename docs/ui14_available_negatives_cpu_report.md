# 实际可用负例策略：CPU 增量报告（2026-09-09）

基于 neg11 分支 `52dde367e7ab6ad819f288c221aa7e0053e64ed9`。用户授权选用较快方案，改为保留全部正例，使用实际有证据的正常图，训练采样 1:1，test 按真实数量评测。原 `e06add6` 对照 checkout 和四卡训练继续保持原状。

## 实现和版本约束

- 默认 `UI14_NEGATIVE_QUOTA_POLICY=available`。独立原图 1:1 缺口不阻止 normalize/finalize/check；仍核对实际正例数和已选正常图数，丢失记录不会被当成允许的缺口。选择数量最多补至原独立图目标，不重复行填数。
- 旧失败 inventory 可由 normalize 直接复用：校验来源/页面/selection 摘要及已选图片身份，保存原 inventory 历史后改策略，不再执行 candidate inventory，不重新扫描未选图片。已验证、属性不变的已选图不重新解码。
- selection 字节、mtime 和页面映射均不因策略迁移改变。扩展清单、snapshot、自己的 normalization_id 和 eval_set_id 绑定新策略。已冻结版本禁止通过默认值切换政策；旧无字段的冻结版本仍为 strict。
- 七个 synth 的 sampler ratio 保持 1.0，其他七项及全局比例保持原值 2.0。实际独立图比例、派生 crop 比例、实际采样比例分别记录；只有单侧标签时按实际样本采样。
- test 保留全部旧正例和实际合格正常图，不重采样。negative_count=0 合法，TP/TN/FP/FN 和 bbox 指标沿用原评分器，不放宽非法输出处理。
- inventory 增加 local 文件名/页面候选发现，结果单独写 `local_candidate_discovery/`。该过程不打开图片、不生成 clean 标签、不改 selection；最快续跑无需额外执行 inventory。
- 未确认 RawImgURL、仅有文件名配对的 local 图仍不自动标负。本次不改变负例证据准入规则。

## 实际执行的验证

| 项目 | 结果 |
|---|---|
| 首轮数据/策略/评分定向回归 | 21 tests，37.591 秒，全部通过 |
| 最终策略、数据、缓存、推理调度、step 0、提交、Excel 回归 | 55 tests，39.316 秒，全部通过 |
| Python AST | 9 个新增/修改 Python 文件通过 |
| `bash -n shell/ui14_neg11_a800.sh` | 通过 |
| A800 正式配置 | CPU 渲染/校验通过，资源/优化参数未变 |
| 本机 A800 数据 normalize/cache/训练提交 | 未执行，没有集群连接 |

首个沙箱测试尝试因临时 SciPy/PyYAML 目录的 Windows 读取权限失败；在可读取这些依赖的普通用户权限下重新执行以上成功回归。依赖仅安装在新 checkout 的临时测试目录，不修改集群环境。

关键回归使用真实 CPU 图片、PNG、数据转换和评分器；GPU detector/模型推理使用测试替身，不构成生产 GPU 验证。完整 CPU 链路包含有独立图缺口、零负例的来源，仍可 finalize/check 得到 ready=true。旧 PNG 路径/分片和旧文件字节、mtime 保持；重复裁图/finalize/normalize 在禁止 Image.open 时通过。旧 checkpoint 配置未改写，完整评测/Excel 仍为 14 项、36 行。

```bash
python -m unittest \
  tests.test_ui14_negative_quota tests.test_ui14_neg11 \
  tests.test_ui14_neg11_eval tests.test_ui14_neg11_raw_audit \
  tests.test_ui14_neg11_detector_reuse tests.test_ui14_inference_workers \
  tests.test_ui14_initial_evaluation tests.test_ui14_submission_resources \
  tests.test_ui14_submit_progress tests.test_ui5_excel_logger
```

## 用户上传的真实数据结果（不冒充本机集群执行）

此前 inventory：合格选中正常图 12,323 张 task/split 独立图（train 11,124、test 1,199），距原独立 1:1 目标缺 83,407。最快恢复直接复用该 selection；cropping/loneword 的完整正常原图数仍为 0，它们原有背景正常裁片保留。实际运行按 selection 当前摘要和文件身份重新核对，不硬编码上述数量。

raw 审计批次 `20260905T222521Z_696fd73b`，父 normalization_id `3588cdd9799c71d8edd15077fa6a487c433c886ebf223f63d6ed57c2e15527cc`。附件报告图片身份复用 68,595、首次检查 0；raw 未自动确认为正常。

用户上传的目录清单包含 local 文件 23,081 条、文件名页面并集 5,712 个。cropping 4,203；occlusion 4,307；radius 3,592；loneword 4,248；large_margin 5,696；inner_margin 353；small_margin 682。70 组来源样例均有同任务页面名匹配，但这是路径/名称统计，未读取对应图片内容，不能当成合格负例配额。

新生产统计、缓存复用数量、GPU 吞吐和任务 ID 尚未产生。默认代码、数据、模型输出路径与分资源执行顺序见 [README](ui14_neg11_README.md)。本次最快顺序为：normalize → CPU cache-prepare → 独立 GPU cache → CPU cache-finalize/finalize/check → submit。

# UI14 alignment context CPU 验证（2026-09-09）

基线：`0379209bd6089e4deeed473d3108f1f41c8d97bf`。
实现位于独立 `codex/ui14-alignment-context-v1` checkout。

实际执行环境为 Windows、Python 3.12、PyTorch 2.8.0+cpu、SciPy 1.16.1、Pillow、
openpyxl 和 PyYAML；没有加载 VLM、detector、GPU 或 A800 数据。
用户提及的旧 Excel 与集群 2000/4000 raw/merged/评分文件在此环境不可用。
因此真实错误分布、正负数量、真实补评指标和集群任务 ID 均尚未获取。
报告中的构造数据验证不冒充历史训练结果。

## 已通过的 CPU 验证

最终整组 48 项测试全部通过（53.600 秒），另通过 19 个变更 Python 文件的 AST 检查、
新 Bash 入口语法及 --help 实际启动验证。包含现有 neg11 评测、原评分计数、14 项/36 行 Excel、
step-0 首评及失败停止、checkpoint-0 恢复、两种资源提交参数、每卡两槽位/
loneword 独占调度，以及以下新增验证：

- 执行生产 generate 方法和实际 logits 约束函数，以 CPU logits 替代 VLM forward。
  slow/hybrid/fast 的合法正例、多框、none、歧义帧和 none 帧回退均通过。
  未完成结构仍抛出失败；没有将非法答案改成合法负例。
- 使用生产 PBD 位置选择器检查实际传给 MTP 的 input_ids：
  box anchor + 5 个 mask 恰好选中六个预测位置；MTP 仍使用原 box resolver、
  coordinate bridge 与 slot 参数。未执行真实模型的 GPU 数值/吞吐验证。
- 捕获并修正一次无 token 被接受时的 AR 接续缓存错误：
  回退重新计算最后一个前缀位置，避免重复该 token 的 KV。
- 对原标签函数生成的合法答案和构造的坏 ref、none/坐标混接、坏坐标、空输出验证语法分类。
  集群 prepare 会再逐条核对现有 ui_alignment 训练 prompt/answer，
  将真实正负记录数和样例保存到 alignment_answer_contract.json。
- 11 张构造图包含非法正/负输出、运行失败、合法 FP、漏检、框未匹配、正确预测和 raw 缺失。
  逐图贡献为 image TP/FP/FN/TN=2/4/4/1，bbox TP/FP/FN=1/5/5；
  其中评分器非法输出正/负各 3 张，贡献 image FP=3、FN=3。
  这些数只属于测试夹具。每图只计一次，成功 gate 旁的历史 error sidecar 不重复计数。
- 直接抽取基线 Git 提交的原 evaluate_merged_file，与重构后的入口对照。
  含非法、合法、空集合和 Figma 开关的全部计数一致，IoU 与 Hungarian 匹配没有改动。
- 审计两 step 的 CSV、JSONL、内嵌图片 HTML 与 GT 点击显示通过。
  缺失 raw 单独列出；重复审计不重新评分；raw/manifest 变化不会误复用。
- 运行既有完整 neg11 CPU 夹具，从真实 PNG、规范化、分片、裁图和标签建立 available 数据，
  包含一个无负例的任务。随后新入口冻结该数据，保持缺口、采样比例、normalization 和 eval_set。
  冻结阶段 Image.open 被设置为直接报错仍成功；第二次 prepare copied=0。
  父目录所有已有文件的字节与文件属性保持一致。
- CPU 模型副本逐文件原子发布、复制时 SHA256、断点续跑、无 optimizer/trainer 状态；
  补评使用独立副本和旧 test。测试用占位权重，不加载张量。
  不同解码身份生成不同目录；重复完整补评跳过 subprocess；
  旧模型文件与状态未被修改，源权重变化被拒绝。
- 正式渲染器输出已通过 YAML 校验：
  A800×4、group_id=2146、aiai_locate 队列、CPT9000、16k、每100步训练记录、
  step0/每1000步全14项、每卡2槽位、EVAL_FAIL_POLICY=stop，以及全部独立路径。

CPU 测试命令（仓库根目录，使用已安装项目依赖的 Python）：

```bash
python -m unittest \
  tests.test_ui14_answer_grammar \
  tests.test_ui14_alignment_audit \
  tests.test_ui14_alignment_context \
  tests.test_ui14_neg11_eval \
  tests.test_ui14_inference_resume \
  tests.test_ui14_run_recovery \
  tests.test_ui14_submission_resources \
  tests.test_ui14_inference_workers \
  tests.test_ui14_initial_evaluation -v
```

本地完成：代码、CPU 测试、参考 YAML 渲染。
集群待执行：真实旧预测审计、真实 neg11 清单冻结、权重 CPU 快照、独立 GPU 补评及正式提交。
上述阶段的可执行命令、成功标志和目录见 [README](ui14_alignment_context_README.md)。

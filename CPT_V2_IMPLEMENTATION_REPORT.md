# UI CPT v2 实施与实验状态（集群前）

日期：2026-08-27。范围：仅 CPT；未改变 SFT 默认行为。

## 当前结论

checkpoint-1549 已证明 UI 任务范式和坐标能力被训练吸收，但现有 10 任务 × 3 样本结果来自
train pool，只能标记为 `train_pool/domain_absorption`，不能证明 held-out 泛化，也不能参与
best checkpoint 选择。已有 formal 在 step 20 的退出不是训练 loss 或 NCCL 根因，而是 UI5
专属的 relation/image_gate/slot_gate 参数更新断言误用于未启用这些模块的 CPT；该检查现已按
真实模块开关隔离。更早的 `DummyOptim.param_groups` 也已兼容。

CPT v2 代码现已具备：图片内容 group 级 98/2 固定切分与泄漏非零退出（多图记录按共享图片
连通分量合并）；train/val/val_fast
recipe 和 manifest；pre/post-MTP 长度及逐条 oversize 明细；全局与 per-task 的
attempted/accepted/trained/skip、main/MTP token、coverage/repeat、同一次 forward 的 token CE；
固定的四种 sampling；十任务专属 evaluator、Base cache、离线重算、task-macro、held-out best
规则和过拟合趋势分析；JSON/JSONL 到严格三 sheet Excel 的可选投影。推理异常现在保留原始
error 且 primary 计 0，不再从指标分母中消失。

本地可验证部分已通过 49 项 CPT 单元测试，包括零泄漏、共享任意图片的多图记录合组、固定
seed、短/超长对账、main+MTP、混合 packed task CE、一对一 box、class-aware defect、VQA、
point parser、rank merge、resume counter、best 规则、过拟合判据及 Excel 非硬依赖。示例
JSON/JSONL/XLSX 为 synthetic/schema-only，不是实验结果。训练侧 Excel writer 已通过严格三表、
冻结表头、筛选和失败不终止训练的测试；当前独立示例 XLSX 尚需重新导出以补上冻结表头和
筛选，因此不能把该文件标记为最终验收通过。

## 尚不能诚实下结论的项目

本地 Windows 环境没有训练集 NAS、CUDA/Magi 和 Merlin，因此以下数值必须由真实集群 job
产生：十任务 train/val rows/groups；实际 sample/main/MTP token share；token-dominant 任务；
oversize 数量与原始/MTP 原因；unique coverage/effective epoch/repeat；A800 与 H20 多卡一致性；
Base、1549、新 checkpoint 的 held-out 曲线；最先出现 train–val gap 的任务；最差 defect
class 和 referring_kg 错误类型；观测开销是否低于 5%。在这些结果产生前，不能声称“没有
过拟合”，也不能指定新 best checkpoint。

## 集群验收与决策

先运行两份 20-step smoke，并从同一 checkpoint resume 至更高 step：

```bash
mlx job submitv2 --path locany_cpt_v4_a100x4_smoke_merlin.yaml
mlx job submitv2 --path locany_cpt_v4_h20x2_smoke_merlin.yaml
```

验收必须确认十任务指标 schema 一致、计数单调不重复、distributed reduce 不死锁、Magi
forward 不接收诊断 kwarg、checkpoint 包含每 rank dataloader state、TorchElastic 显示原始
rank/file/line，并与 Milestone 0 冻结代码的同配置短跑比较 step-time 增幅。

随后跑完整静态长度分析、四策略离线模拟、Base/候选 checkpoint 的 val_fast teacher-forced
与 milestone generation，最终候选再跑完整 2% held-out。至少三个 milestone 后运行
`analyze_locany_cpt_curves.py`，才能判断哪个任务先过拟合。

当前安全默认保持 `sample_equal`，token limit 保持 A800 7268、H20 8192，不截断 ground
truth，也不自动扩上限。这不是最终策略结论，而是冻结基线。若真实结果同时显示大任务覆盖
不足、小任务 repeat 高且 held-out 停滞，下一轮优先比较 `hybrid`，再比较 `sqrt_size`；仅在
token share 极端失衡时测试 `token_balanced`。所有策略按相同有效 sample exposure 比较。
Stage A hybrid/sqrt-size + Stage B 短程 sample-equal 只在上述证据成立后启用，本次合入不默认
开启。

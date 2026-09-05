# UI14 CPU 检查报告

日期：2026-09-06。基线 `10efee4461418b952a2bf2dfc6d3e7032c591a0e`。
交付分支：`codex/m32-cpt9000-ui14-v1`。

**本地代码与连接验证通过；A800 实际数据、detector 缓存和训练尚未执行。**
本地为 Windows CPU 环境，未获得 A800 可用连接，也无法读取其 `/mnt/bn/...`。因此不填造 UI9 样本规模、真实 train/test 重复数、缓存完成率或训练指标。

## 已执行

184 项 CPU 测试通过（183 项回归集 + 1 项 processor 序列化专项）。Python 3.12，PyTorch `2.14.0+cpu`，`torch.cuda.is_available() == False`。模型测试仅构建 hidden=32 的 relation/task bank，不加载 3B 主干。三个改动相关 Shell 脚本均通过 Git Bash `bash -n`。

| 检查 | 本地结果与覆盖边界 |
|---|---|
| 14 项注册与编号 | ID 0–13 固定；旧 class_id 映射不变；4 个 relation family；全图仅 4/5/11 |
| 相同提示词隔离 | cropping/synth_cropping、occlusion/synth_occlusion、两个换行来源按显式 ID 分离；冲突 ID 拒绝 |
| 375 画布转换 | 真实样例 `941×2048`、`[183,449,269,479]` → norm1000 `[488,550,717,587]`；Y 同样乘 W/375 |
| 多框与来源 | Location 列表/对象、字段优先级、多框、objects.bbox_type=real；正常参考图不生成负样本 |
| crop 映回原图 | 像素偏移往返与 norm1000 量化误差验证；裁剪位置只读 detector；邻接文本组不被切开 |
| 固定 train/test | 九项共 18 份 fixture 输入逐文件摘要不变；派生 split 继承；detector 输入仅包含 image |
| 原图重复 | fixture 中跨路径、跨来源、跨 train/test 的相同 RGB 图被识别；无重新切分/过滤 |
| 七项 crop 缓存 | 固定模拟 detector 输出，运行真实 CPU v5 几何、gallery、ready marker 和完整 validator；共 14 份 train/test 缓存；篡改输入后验证失败 |
| 联合 recipe | 实际调用 normalize/finalize/check 连通 14 路；权重每项 1/14；只引用 train；旧排除/人工修复保留接口验证 |
| 源图采样 | crop 数量不同的原图近似等抽取；轮换覆盖；2:1 负正比；单侧任务无伪造另一侧；固定 seed/epoch 可重放 |
| checkpoint/processor | 14 项 embedding/expert/head 小模型保存重载；patch 输出独立注册表模块和 processor metadata；真实 ProcessorMixin 序列化恢复任务表 |
| M32 损失兼容 | 旧五项 focal loss 与原参数数组计算逐位相同；既有 relation/M32 CPU 回归通过 |
| 14 项评测 | 每项 worker 使用各自 view/cache/Figma 策略；原图评分复用真实 Hungarian IoU=.1/非法输出函数；bbox 无 TN；negative_count=0 可表示 |
| 断点续评 | 模拟 worker 执行、实际状态/历史/Excel 写入；已有 UI5 或缺一项 UI9 不算完成，补齐后只有一轮结果 |
| Excel/best | 每轮 36 行；five_task 与 ui9 独立 macro/micro；best 保持 UI5 macro；两个 sheet 名称兼容 |
| 正式 YAML | 实际渲染并用 PyYAML 解析；单节点 A800_SXM_40GB ×4、group2146、指定队列、CPT9000、SFT0、16k、7268/12800、GA2、双 LR 参数核对 |

用于复现的 CPU 测试命令（在具备项目 CPU 依赖的 Python 环境中）：

```bash
python -m unittest \
  tests.test_ui14_pipeline tests.test_ui14_model \
  tests.test_ui_defect_data tests.test_ui_relation_pipeline tests.test_relation_modules \
  tests.test_ui5_excel_logger tests.test_ui5_sampling_coverage tests.test_ui5_pipeline \
  tests.test_ui5_eval_detector_scan_v5 tests.test_ui5_tiled_evaluation \
  tests.test_ui5_eval_detector_scan tests.test_validate_ui5_crop_training_ready
bash -n shell/ui14_cpt9000_a800.sh
bash -n shell/run_locany_ui5_pipeline.sh
bash -n shell/train_locany_ui_defect.sh
```

fixture 集成测试中只有旧 UI5 审核 marker、旧 1555 图 cache 以及 GPU worker 执行使用 mock；新标注转换、14 份 crop 几何/cache validator、recipe、评分、Excel 与报告绑定均实际运行。processor 测试提取原类以避开视频/LMDB 可选依赖，使用真实 Hugging Face ProcessorMixin 序列化。

## 必须在集群执行的检查

`bash shell/ui14_cpt9000_a800.sh normalize` 读取实际落地 manifest 与 18 份输入，验证每张输入图可读并记录尺寸/像素身份/原 JSONL 摘要。

`cache` 单独使用 GPU 运行 PP-OCRv5 与 OmniParser，产出七任务 × train/test 的完整检测、裁剪计划和 crop 文件。本地测试中的固定 detector 结果不用于正式训练。

`finalize` 读取真实旧审核 marker、排除清单、旧 1555 图 cache 与新 14 份 cache，验证源图未变、crop PNG 摘要、split/路由/提示词/坐标连接、评测全量任务及 checkpoint 权重分片可读性，输出：

```text
/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_data/ui14_cpt9000_v1/cpu_check_report.json
/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_data/ui14_cpt9000_v1/formal_job.yaml
```

真实统计写入同目录 normalization_stats、sampling_stats、image_overlap 和各 coverage 报告。GT 跨 crop 与冻结数据重复均如实记录，不用 GT 改动测试 crop，也不改动固定 split。提交及训练启动会拒绝失败或摘要失配的报告。

`prepare_ui9_datasets.py v2` 未出现在本地仓库，未能对其源码逐行比对；使用了请求中指定的转换契约及既有准备任务提供的真实样例。A800 manifest/schema 的实际适配结果必须以 normalize 命令产物为准。

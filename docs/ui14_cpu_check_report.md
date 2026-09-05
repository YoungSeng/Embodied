# UI14 repair v2 CPU 检查报告

日期：2026-09-06。增量基线：`5d1e07bd4c1eb347b094d2463570ec9249cccf56`。
分支：`codex/m32-cpt9000-ui14-v1`。

**本地代码回归已通过；实际 UI9 数据扫描、GPU detector 缓存和 mlx 提交由远程开发机执行。**
用户确认该环境不能从本地连接、代码只进不出。本报告不填写真实样本数、实际解析影响数量、缓存完成率或任务 ID；这些值由随代码交付的四步入口写入正式派生目录。

## 已执行的 CPU 验证

109 项 CPU 回归通过。Python 3.12.14，PyTorch 2.14.0+cpu，CUDA 不可用。没有加载 3B 模型、运行训练或调用 GPU detector。另核对 14 个修改/新增 Python 文件的 AST，三个正式 Shell 入口通过 `bash -n`。

附件 `prepare_ui9_datasets.py v2.1` 的 SHA-256：
`72a43e551c4388fdeaef985a9206105da400e4351bc817ef4d7cdedbce3c084e`。

共享解析模块的 14 个 CPU 函数/类与附件逐项 AST 一致，路径映射常量也保持一致；审计用旧消费器的 7 个函数与基线 AST 一致。原样提取不包含 repair/copy/split 执行入口，不依赖 fcntl 或 GPU。

| 检查 | 已执行结果与边界 |
|---|---|
| GT 字段优先级 | rect_err_1/rect_err1、大小写、多框、空高优先级回退、mbr/shift/combine/rectN 均按 v2.1 |
| 存储格式 | 两点矩形、Objects 单对象/列表、Location 直接坐标/JSON 字符串/bbox 包装字典，构造回归通过 |
| 主图与负样本 | ScreenShotURL 嵌套字段、标注集 messages 图片、主图唯一性；只有已有 Objects=[] 才能以 LocalImgURL 作输入，RawImgURL 不作为替代主图 |
| 坐标 | 合成 X/Y 同乘 W/375；941×2048 示例输出 [488,550,717,587]；crop 偏移往返通过；标注集按声明的 real/norm1/norm1000/canvas/scale/raw-export 转换，不按数值猜尺度 |
| 修复后标注 | 不再次裁框，完全保留选中多框；已修复坐标若仍越界则报告并拒绝产物 ready |
| 当前文件 split | 18 份构造输入不修改；七类模拟移动的 14 条记录保留来源 split，规范化 split 按实际文件；备份与隔离文件不读取 |
| 修复批次与实际数量 | 构造验证 manifest/repair_summary run_id、修复检查 PASS、after_train/test 和实际 JSONL 数量；未完成发布、批次失配或数量失配会失败 |
| 解析影响统计 | 将真实格式频次、旧解析失败、旧 split/ID 接入拒绝，以及双边成功后的解析差异分别计数；构造样例验证统计归类，不推断真实错误标签数 |
| 页面与原图重复 | 九来源共享页面按文件 split 统计；七合成来源跨页面 train/test 泄漏被报告并拒绝，不自行重分；跨路径相同 RGB 图片仍报告 |
| 缓存与标签刷新 | 真实 CPU v5 几何和 validator 处理固定 detector 输出，14 份任务/split 缓存通过；修改 GT 后保留原计划摘要但刷新标签和 repair 完成标记；旧图片清单索引被拒绝 |
| recipe/评测/Excel | 14 路等权、源图轮换、单侧标签与 2:1 双侧比例；完整 14 项原图评测、UI5 独立 best/macro、两个 sheet 与断点补项的既有回归通过 |
| 正式运行绑定 | source_snapshot、源文件、normalized/derived/cache 和报告摘要绑定；不同 repair 的 optimizer 输出目录不能互相恢复；finalize 失败仍保留前一步扫描统计 |
| 正式 YAML | 用既有渲染入口生成并解析 YAML，核对单 worker A800_SXM_40GB×4、group2146、指定 queue、CPT9000、16k、1k 评测、4k 保存；EVAL_FAIL_POLICY=stop；显式联合 META_PATH 指向 repair_v2 |

与基线正式 environment 对比，只有数据版本 DATA_VERSION 和运行标识 RUN_NAME 更新；输出目录改为独立 repair-v2。任务编号、采样、模型、损失和训练参数保持基线值。

可复现命令（在具备项目 CPU 测试依赖的 Python 环境）：

```bash
python -m unittest \
  tests.test_ui14_repair tests.test_ui14_pipeline tests.test_ui_defect_data \
  tests.test_ui5_pipeline tests.test_ui5_excel_logger \
  tests.test_ui5_eval_detector_scan_v5 tests.test_ui5_tiled_evaluation
bash -n shell/ui14_cpt9000_a800.sh
bash -n shell/train_locany_ui_defect.sh
bash -n shell/run_locany_ui5_pipeline.sh
```

新增 repair 回归 14 项。集成 fixture 的旧 UI5 审核 marker/1555 图 cache 和 GPU worker 执行使用 mock；新标注转换、缓存几何、计划/标签绑定、recipe、原图评分、Excel 和报告校验实际执行。固定 detector fixture 不进入正式数据目录。

## 实际数据统计的产生位置

派生根目录：
`/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_data/ui14_cpt9000_repair_v2`。

| 远端阶段 | 将产生的实际证据 |
|---|---|
| normalize（CPU） | 当前 manifest、repair_summary、18 份 JSONL 和图片可读性；source_snapshot.json、九项规范化 train/test、normalization_stats.json、ui9_page_split.json、ui9_image_overlap.json、parser_compatibility_issues.jsonl；cpu_check_report.json 的 normalization_complete，ready=false |
| cache（GPU） | 仅新增七项 crop 任务的 train/test detector 与横向计划；派生标签、crop PNG、ui14_label_cache_ready.json。周期评测不运行 detector |
| finalize（CPU） | 复用旧 UI5 审核 recipe/test cache；生成 training_recipe.json、evaluation_manifest.json、14 项连接检查、sampling_stats、完整 image_overlap、formal_job.yaml/formal_runtime.json；全部通过后 cpu_check_report.ready=true |
| submit | 摘要校验通过后执行既有 mlx job submitv2。任务 ID 以远端 mlx 返回为准，本地未提交 |

报告的 post_repair_sources（normalize 时为 tasks）含每份文件的实际记录数、正负数、格式数量、GT 字段及修复数量。parser_comparison 分开给出 legacy_parse_failure_records、legacy_consumer_failure_records、parse_result_difference_records；页面统计覆盖 UI9 跨来源的 train/test 归属。完整字段解释和四步命令见 [运行文档](ui14_cpt9000_a800.md)。

参考 YAML 已提交为 [locany_m32_cpt9000_ui14_a800x4.yaml](../jobs/rendered/locany_m32_cpt9000_ui14_a800x4.yaml)。远端 finalize 会用同一渲染函数在派生根目录产生本次实际提交 YAML，并将其摘要绑定到 CPU 报告。

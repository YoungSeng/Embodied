# UI14 repair v2 CPU 检查报告

日期：2026-09-06。增量基线：`5d1e07bd4c1eb347b094d2463570ec9249cccf56`。
分支：`codex/m32-cpt9000-ui14-v1`。

**本地代码回归已通过；实际 UI9 数据扫描、GPU detector 缓存和 mlx 提交由远程开发机执行。**
用户确认该环境不能从本地连接、代码只进不出。本报告不填写真实样本数、实际解析影响数量、缓存完成率或任务 ID；这些值由随代码交付的分阶段入口写入正式派生目录。

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

## 准备阶段进度增量验证

在 repair 提交 `46cdd657c6e79c937b3ea452de223e4d17c72de2` 上补充 normalize/cache/finalize 进度。此次 50 项 CPU 回归通过，其中新增进度回归 10 项；8 个修改/新增 Python 文件 AST、正式准备 Shell 的 `bash -n` 和两个 Python 入口的 `--help` 均通过。没有运行 GPU detector 或训练。

- 按已完成工作计算速度/ETA；无完成量时不虚构剩余时间，异质任务外层不按任务数推算整条命令 ETA。
- 前台未推进计数时后台仍更新状态；异常和中断保留失败语义，日志写入失败不覆盖数据处理异常。
- normalize 开启/关闭进度的产物逐文件摘要相同；同一 detector 计划生成的 crop 标签与完成标记逐字节相同。
- cache 仍调用原 detector 入口，按原顺序处理七任务 × train/test 共 14 项，保留四卡、resume、v5 几何参数；GPU 子进程失败会传播，后续标签任务不会启动（worker 使用 mock）。
- detector 状态忽略前次运行记录，并同步本次 worker ETA；完整 finalize 产出仍通过数据绑定检查，进度文件不进入 artifact_digests。
- 原 14 项评测、UI5 best/Excel、正式 YAML 和 `EVAL_FAIL_POLICY=stop` 的既有回归继续通过。

```bash
python -m unittest tests.test_ui14_progress tests.test_ui14_repair \
  tests.test_ui14_pipeline tests.test_ui5_eval_detector_scan_v5
bash -n shell/ui14_cpt9000_a800.sh
```

## 每卡双推理进程增量验证

基于 `c404c15440e27a87f81603cd37532f3405f99296`，正式 UI14 profile 改为 `EVAL_INFERENCE_WORKERS_PER_GPU=2`，四卡共 8 个独立推理进程槽位。65 项 CPU 回归通过，其中新增并发回归 3 项；8 个 Python 文件 AST 和两个正式 Shell 入口语法检查通过。

- 使用真实 CPU 子进程和共享到达屏障验证首批 8 个进程同时运行，每个 GPU 环境标识对应 2 个进程；14 项任务全部且仅执行一次，日志和汇总独立。
- 模拟首批 worker 失败，调度器返回失败并保留未启动任务；周期评测仍使用 `EVAL_FAIL_POLICY=stop`。
- 参数贯穿 profile、YAML、运行配置、Shell、评测入口和调度器；UI14 正式值固定 2，旧 UI5 默认值保持 1。状态记录 worker_id/worker_slot，便于识别同卡进程。
- 使用现有入口重新渲染正式 YAML，与上一版仅相差新增的 `EVAL_INFERENCE_WORKERS_PER_GPU: "2"`；训练四卡/16k、UI5 指标口径、best 保留及 detector 缓存并发保持原值。

```bash
python -m unittest tests.test_ui14_inference_workers tests.test_ui14_pipeline tests.test_ui5_pipeline
```

未连接 A800，也未加载 GPU 模型；CPU 并发检查不代表实际显存占用测量。已有完整产物时，更新代码后执行 `prepare_ui14_sft.py --stage check`，刷新正式 YAML 和报告绑定摘要，再沿用 submit 入口。

## EXIF=0 与 normalize 续跑增量验证

增量基线：`9c4de2f000520e5d73f9dd85deaf7f3c9b8ea695`。46 项相关 CPU 回归通过，其中针对本次故障新增 8 项；5 个 Python 文件 AST 和正式准备入口的 bash 语法检查通过。另使用该提交的原版 normalize 函数生成构造产物，直接验证旧版本兼容性，并单独复查 finalize/check 保留续跑计数。测试数据均为本地构造数据，没有加载 GPU 模型。

- EXIF Orientation=0 可转换，原文件摘要、解码 RGB 像素、尺寸、source_image_id 与 bbox 均不变；2–8 保留原有拒绝策略。
- 旧整体 complete=false 且对齐 split 含部分成功记录时，复用其余 16 个 split，只重建 ui_alignment/train、test。来源 snapshot/normalization ID 保持一致，成功 split 的三份数据产物摘要不变，原解析差异明细保留。
- 对复用路径禁用 Image.open、inspect_image、image_identity 和源记录重新解析，仍可成功；再次 normalize 为全部 18 split 复用。
- 补完对齐后中断，仍能迁移旧全局报告中的其他 16 split；产物已写但未写完成标记时，下次只信任此前独立完成的 split。
- 单个 detector input 摘要失配只重建对应 split；来源 snapshot 变化不会复用旧 normalization ID；全局页面冲突在全部复用时仍会检查并报告失败。
- normalize 的 reused/rebuilt 计数在最终 finalize/check CPU 报告中保留。14 项注册、375 坐标投影、crop 缓存连接、CPT-9000、每 GPU 两个评测进程、EVAL_FAIL_POLICY=stop、周期评测/Excel 与正式 YAML 既有回归通过。

原版 `9c4de2f` 生产器的独立兼容性检查结果（20 条构造记录，非真实 UI9 数据）：

| 检查 | 实际本地结果 |
|---|---|
| 原版 normalize | 成功 18、EXIF=0 失败 2，整体 complete=false |
| 新版首次续跑 | reused=16 split / 16 条；rebuilt=2 split / 4 条；成功 20、失败 0 |
| 再次 normalize | reused=18 split / 20 条；rebuilt=0；复用路径原图读取次数=0 |
| 来源绑定 | normalization ID 不变；旧成功 split 的标注与 detector input 摘要不变 |

```bash
python -m unittest tests.test_ui14_normalize_resume tests.test_ui14_repair \
  tests.test_ui14_progress tests.test_ui14_pipeline
```

**真实集群 normalize/cache/finalize/submit 均未在本地执行，未产生提交任务 ID。** 用户提供的原运行统计为 130,567 条中成功 120,010、EXIF=0 失败 10,557。由该统计计算，本次目标为复用 16 split / 111,007 条，重建对齐 train 17,604 条、test 1,956 条（合计 19,560），最终 normalized=130,567、failed=0；这些是远端验收目标，不是本地实测结果。实际续跑数量由远端 `cpu_check_report.json.normalization_resume` 和进度日志给出。

## cache 的 CPU/GPU 阶段拆分与并行续跑

增量基线：`cb8d67d6cc3496ec95c3fda8533bb389e7caf870`。87 项 CPU 回归全部通过（88.892 秒），包含新增的 10 项缓存阶段回归；6 个 Python 文件 AST、Shell `bash -n`、新入口 `--help` 和 `git diff --check` 通过。正式配置及已提交 YAML 没有改动。本地只执行 CPU 构造验证，没有执行真实集群 detector 或训练，也没有测量真实数据吞吐提升倍数。

- cache-prepare 默认 16 线程，受控并发验证实际使用多个线程；原图指纹和方向后尺寸与旧消费者一致（覆盖 EXIF 0/1/6）。旧清单、分片及已完成 detector 结果的摘要逐项保持一致，旧 GPU 分片仍可通过 resume 校验。
- 图片 journal 落盘后重新打开可全部复用，禁用图片打开和重新哈希仍通过；文件变化只重新扫描对应图片；半截末行会被安全丢弃，已完成记录保留，异常退出后可继续。
- CPU prepare 不要求 Paddle、icon Python 或 parser 目录存在，不启动任何 detector 子进程。14 个 split 的 CPU 构造结果：首次 scanned=14，再次 reused=14/scanned=0（非真实 UI9 数量）。
- GPU 调度只派发全部 14 split 的 text 和 icon，共 28 个阶段调用。主进程禁用原图打开、原图 stat、prepare 和裁剪函数仍通过；真实 GPU worker 在该检查中使用 mock。缺少最后一个 split、分片摘要变化或误用旧五任务注册表时，在任何 GPU 子进程启动前失败。
- GPU 子进程失败仍停止；所有分片完成但 stage_summary 缺失时，可以不加载 GPU 模型而补齐汇总。
- CPU 收尾只派发 merge/crop 和任务标签。使用固定检测框，通过真实 CPU 子进程为 synth_cropping 的构造 train/test 完成合并、几何计划、crop PNG、覆盖统计和标签 marker；无需 Paddle Python。
- 原 EXIF=0/normalize 续跑、375 投影、UI14 recipe/评分/评测补齐/Excel、UI5 几何与正式 YAML 回归均通过；CPT-9000、4×A800、两评测 worker/GPU 和 EVAL_FAIL_POLICY=stop 保持原值。

```bash
python -m unittest tests.test_ui14_cache_stages tests.test_ui14_progress \
  tests.test_ui14_normalize_resume tests.test_ui14_repair \
  tests.test_ui14_pipeline tests.test_ui5_eval_detector_scan
```

当前旧 cache 进程需结束后再更新代码和运行新入口，同一输出目录不能并发写。已完成 normalize 不重跑；先在 CPU 完成 `cache-prepare`，再申请四卡 A800 运行 `cache`，检测完成释放调试 GPU，最后在 CPU 上运行 `cache-finalize` 和 `finalize`。第一次升级仍需建立图片 journal，已有 GPU 缓存继续复用。迁移命令见 [运行文档](ui14_cpt9000_a800.md)。

## 2026-09-07 并行裁图、原图级续跑与快速 finalize

本增量基线 `75fc54a67f1ce07fa19c5e93db5611000679285a`。本地执行 111 项 CPU 回归，全部通过，耗时 72.633 秒；13 个修改/新增 Python 文件 AST、Shell `bash -n`、`git diff --check` 通过。使用现有正式 profile 重新渲染 YAML，与已提交 `jobs/rendered/locany_m32_cpt9000_ui14_a800x4.yaml` 解析后的内容完全一致。CPT-9000、4×A800、16k SFT、14 项评测、两评测 worker/GPU、EVAL_FAIL_POLICY=stop 均保持原值。

实际执行范围是本地 CPU 构造数据及固定 detector 输出上的回归，包括真实 PNG 写入、CPU merge/crop 子进程、finalize/check 和 YAML 渲染。未连接开发机/A800，未执行真实 130,567 条数据的准备、GPU 检测或 mlx 提交；未测量挂载盘吞吐、旧约 950 张半成品的实际复用量或真实剩余 ETA，没有任务 ID。

| 验证 | 实际结果 |
|---|---|
| 并行及顺序 | 3 张唯一原图、4 条记录（含重复原图记录），实际 2 个 worker；输出 8 条 crop 记录，原记录/tile 顺序稳定 |
| 旧半成品接入 | 构造 1 张原图对应的 2 个旧 PNG，首次 migrated=1、built=2（4 个新 PNG）；旧 PNG 字节及摘要保持不变 |
| 全部复用 | 再次运行 reused=3、built=0；禁用 Image.open 和 PNG 编码、改变压缩级别仍可成功 |
| 像素/GT | 新旧解码 RGB 像素、bbox、定位输出、正负标签与逐记录覆盖统计一致；包含跨 tile GT 和 EXIF=0 |
| 中断与坏文件 | 完成记录在异常后保留；截断 journal 末行可恢复；损坏一个 PNG 只补写该 PNG，其他 2 张原图复用 |
| 局部失效 | 仅改标注不读图片；改一张原图的几何只重建该原图；真实截图内容与 normalized 身份不符时停止，规范化/计划对应更新后只重建该图 |
| CPU 准备 | EXIF 0–8 尺寸检查禁止整图解码/转置仍通过；完整 split 不重新准备原图，已有 detector ID 和分片保留 |
| GPU 分片 | 构造 2 个已完成分片，更新单图后仅其所在分片失效，另一分片的字节和 mtime 不变且 resume 校验通过；GPU 入口仍只派发 text/icon |
| finalize/check | 完整 14 任务构造链路通过；重复 finalize 与启动前校验禁止图片打开、禁止裁图仍成功；独立全量复核读取像素并拒绝坏 crop |
| 完成计量 | 500 张 built 采样门槛不计 migrated/reused，保留各 split 的 pending/ETA；性能采样不会停止正式准备 |
| 单协调进程 | 同一输出根目录第二个准备进程不能获取锁 |

验证命令（CPU）：

```bash
python -m unittest tests.test_ui14_parallel_crops tests.test_ui14_cache_stages \
  tests.test_ui14_progress tests.test_ui14_normalize_resume tests.test_ui14_repair \
  tests.test_ui14_pipeline tests.test_ui5_eval_detector_scan \
  tests.test_ui5_eval_detector_scan_v5 tests.test_ui14_inference_workers
bash -n shell/ui14_cpt9000_a800.sh
```

修改文件：

- `scripts/ui14_crop_materialization.py`（新增）：有界线程池、低压缩 PNG、原图完成索引、标签绑定、旧 PNG 接入、实际新裁图计量。
- `scripts/ui14_verification.py`（新增）：按文件属性绑定的验证 journal、首次并行内容检查、结构验证记录、准备锁。
- `scripts/prepare_ui14_sft.py`：并行裁图接入，finalize 只消费完成产物，增量 check、`--full-verify` 与图片证据绑定。
- `scripts/prepare_ui14_detector_crops.py`、`scripts/ui14_cache_prepare.py`、`scripts/prepare_ui5_eval_detector_crops.py`：已完成准备/几何复用，图片头读取和局部分片更新。
- `scripts/ui14_common.py`、`scripts/ui5_eval_detector_cache.py`、`scripts/ui14_profile.py`：摘要/结构检查复用及提交、训练启动前验证。
- `shell/ui14_cpt9000_a800.sh`：UI14_CROP_WORKERS、UI14_PNG_COMPRESS_LEVEL、check-full。
- `tests/test_ui14_parallel_crops.py`（新增）以及 `tests/test_ui14_cache_stages.py`、`tests/test_ui14_pipeline.py`、`tests/test_ui14_repair.py`：回归与更新后的阶段契约。
- 本报告及 `docs/ui14_cpt9000_a800.md`：分资源运行命令、迁移和计量说明。

真实运行会在现有派生根目录保存 `crop_performance/first_500_built.json`、`crop_performance/latest.json` 和每次 run_id 的快照。报告记录新原图/s、crop/s、实际并发、读取/解码/编码/写入时间、CPU/RSS、逐 split 复用/新建/待处理数量与裁图 ETA。0.79 原图/s 是用户给出的旧日志基线；8 原图/s 是目标，不能用本地小图回归速度宣称达到。无需新增 smoke 训练。

## 2026-09-07 准备完成标记发布失败的恢复修复

增量基线 `1d6f6c147711d3eb22e9014d945fc0d54ffb1c4e`。用户远端日志显示 synth_occlusion/train 已完成 56,000 task-images、53,047 张内容唯一图片及 71 个分片，随后在 `publish_prepared` 核验 unique_images.jsonl 时发现属性变化并退出。这些数量来自用户日志，本地未读取该集群目录；旧日志没有 before/after 属性值，不能确认具体变化原因。

修复文件为 `scripts/ui14_verification.py`、`scripts/ui14_cache_prepare.py` 和 `scripts/prepare_ui14_detector_crops.py`。SHA256 对不稳定的同一文件最多读取 4 次，不缓存失败尝试的摘要；核对路径和文件描述符各自前后属性、大小/mtime、读取字节数及设备/inode。路径 stat 和描述符 fstat 的 ctime 各自在同一种 API 内比较，兼容本地 Windows 运行时的不同返回语义；不取消 ctime 变化检查。持续变化仍报错并展示前后属性。

缺少最后 ready 标记时，CPU 恢复核对来源摘要、唯一图片、任务样本顺序、全部分片成员及图片 journal，再只补发布标记。正常完成的 split 仍 reused；恢复的 split 输出 `[cache-prepare recovered]`。缺分片、错误 split/selection 等不会被当作完成，持续不稳定的读取不会触发整份清单重建。

新增 9 项 CPU 回归，连同现有回归共 **120 项全部通过，78.210 秒**；5 个 Python 文件 AST 和 `git diff --check` 通过。

- 短暂属性变化只重读对应文件，稳定后才保存摘要；下一次可复用。
- hash 后、保存验证记录前发生真实内容变化，旧摘要不会写入 journal。
- 连续 4 次变化仍失败，缺文件不做无效重试；同尺寸/时间戳下的文件替换由设备/inode 检查拒绝。
- stat/fstat 的固定 ctime 差异不会误报；描述符自身 ctime 在读取期间变化仍被拒绝。
- 构造完整准备产物后删除最后标记，禁止打开原图、禁止重建清单仍可恢复；manifest、分片与已有 detector 完成文件的字节和 mtime 不变。
- 截断分片及过时 selection 均不能补发成功标记。

```bash
python -m unittest tests.test_ui14_verification_retry tests.test_ui14_parallel_crops \
  tests.test_ui14_cache_stages tests.test_ui14_progress tests.test_ui14_normalize_resume \
  tests.test_ui14_repair tests.test_ui14_pipeline tests.test_ui5_eval_detector_scan \
  tests.test_ui5_eval_detector_scan_v5 tests.test_ui14_inference_workers
```

本次仅执行本地 CPU 验证，未执行真实集群恢复、GPU 检测或提交。数据、裁剪/模型策略与正式训练配置未变；更新代码后在原输出目录重新运行 cache-prepare，不需要重跑 normalize 或删除缓存。

## 2026-09-07 GPU detector 每卡多进程与分片续跑

增量基线 `670d6701f2f136b97792b0921fa74385d2476ac6`。UI14 cache 默认每 GPU 4 个 detector worker，`UI14_DETECTOR_WORKERS_PER_GPU=5` 使用每 GPU 5 个；text 与 icon 均适用。旧 UI5 独立 detector 默认每卡 1 个，训练周期评测保持每卡 2 个推理 worker。四卡正式训练配置、YAML、GT-free 几何及数据身份算法均未修改。

本地 CPU 回归 **125 项全部通过，96.261 秒**；5 个 Python 文件 AST、Bash 入口语法与 `git diff --check` 通过。本轮新增 5 个测试方法，其中多组参数化构造分别验证 4/5 进程和 text/icon；阶段交接测试同步扩展。

修改文件：

- `scripts/prepare_ui14_detector_crops.py`、`shell/ui14_cpt9000_a800.sh`：环境变量/CLI 接入，GPU 阶段传递并发及每 worker 两个图片加载线程；CPU handoff 的配置不变。
- `scripts/prepare_ui5_eval_detector_crops.py`、`scripts/run_ui5_crop_audit.py`：1–5 进程参数及子进程转发、延迟模型加载、有界图片预加载、退出时回收同轮 worker、清理过时进度、新增图片吞吐统计。
- `tests/test_ui14_detector_workers.py`、`tests/test_ui14_cache_stages.py`：CPU 构造测试及完整 UI14 阶段交接回归。
- 本报告与 `docs/ui14_cpt9000_a800.md`：并发调整、分片复用和停止后续跑命令。

验证覆盖：

- 每卡 4/5 个进程，四卡分配 16/20 个互斥 worker 索引；PP-OCRv5 和 OmniParser 的命令保持各自 Python 环境。旧双进程参数仍兼容。
- 每组构造 27 张小图和 27 个分片，预先完成 3 个，另留 1 个未完成输出；新增 24 张恰好各检测一次。text/icon × 4/5 进程四组均通过。再次改变并发后全部复用，禁止加载模型和原图仍通过。
- 已完成分片、标记、输入 manifest、分片成员和 detector_config 的文件字节与 mtime 保持不变；已有完整 stage_summary 不重写。以上是 CPU 构造复用数量，不是集群真实缓存统计。
- GPU 阶段仍只调度全部 14 个 split 的 text 和 icon；将并发 4 改为 5 不要求重做 cache-prepare，禁止扫描/解码原图仍通过交接检查。
- 旧进度文件不能计入新一轮 ETA；新阶段汇总 `reused_images=3`、`new_images=24`，吞吐只用本轮 24 张除以本轮耗时。
- 构造 worker 失败和启动中断，其他已启动 worker 均收到 terminate 并回收，完成分片保持原样，无成功 summary。
- 两线程预加载最多持有 4 张图片，保持 100 张输入的顺序；关闭生成器后所有未消费图片被释放。

```bash
python -m unittest tests.test_ui14_detector_workers tests.test_ui14_verification_retry \
  tests.test_ui14_parallel_crops tests.test_ui14_cache_stages tests.test_ui14_progress \
  tests.test_ui14_normalize_resume tests.test_ui14_repair tests.test_ui14_pipeline \
  tests.test_ui5_eval_detector_scan tests.test_ui5_eval_detector_scan_v5 \
  tests.test_ui14_inference_workers
bash -n shell/ui14_cpt9000_a800.sh
```

用户日志的 3,000/53,047、19.81 images/s、约 42 分钟 ETA 属于 synth_occlusion/train 的旧并发 text 阶段；本地未连接 A800，没有每卡 4/5 个进程的真实吞吐、峰值显存或全量剩余时间测量，也未启动训练提交。远端完整分片继续复用，未写 `.done.json` 的当前分片需重算（默认最多 750 张/片）。实际复用数量和新吞吐由续跑日志及各 split 的 `detections/{text,icon}/stage_summary.json` 给出。

## 2026-09-07 小 split 与零散剩余分片的 GPU 分配修复

增量基线 `36b161fb95b7b5aed502bce10f66881d090189e4`。用户日志为 synth_occlusion/test text，5,907 张图片。按现有 750 张/片共 8 个分片；上一版按 GPU 分组排列 16 个 worker 槽位，前 8 个 worker 对应 GPU 0/1，所以 GPU 2/3 无任务。该原因已从代码确认；本地未读取远端 GPU 进程。

本次相关 CPU 回归 **54 项全部通过，20.289 秒**，3 个 Python 文件 AST 和 `git diff --check` 通过。验证命令：

```bash
python -m unittest tests.test_ui14_detector_workers tests.test_ui5_eval_detector_scan_v5 \
  tests.test_ui5_eval_detector_scan tests.test_ui14_inference_workers
```

`scripts/run_ui5_crop_audit.py` 改为父进程统一计算 pending 分片快照，按 GPU 轮流分配活跃 worker，只启动有任务的进程；`scripts/prepare_ui5_eval_detector_crops.py` 和旧 audit 入口接收显式 `--assigned-shard` 名单。未完成分片不再由各 worker 独立过滤/重新编号，完成其他 worker 的任务不会导致漏项或重复。原有直接 worker 调用仍支持旧模运算分配。分片、detector 配置、完成标记格式保持兼容；正常续跑只传递原分片文件名。旧完整 summary 不改写，新 summary 记录每卡实际 worker 数和各 worker 的分片分配。

`tests/test_ui14_detector_workers.py` 新增 3 个 CPU 回归方法：

- 每卡上限 4/5 × text/icon × 8 个待处理分片/零散续跑共 8 组构造。8 片时四卡各 2 个 worker；原编号 0/4/16/20 的 4 个剩余分片四卡各 1 个 worker。每张待处理图片恰好检测一次，完成分片、manifest、配置的字节和 mtime 均不变。
- 对 0/1/2/3/4/8/16/19/23/71 个分片以及每卡 1/4/5 上限检查 GPU 轮换、非连续 GPU 编号、进程上限和分片集合无缺失无重复。新旧子进程 CLI 均传递显式名单。
- 重复、越界路径和未知分片名在加载模型前拒绝。

CPU 构造图片不用于声明 A800 性能。真实续跑日志将显示 `pending_shards`、各卡进程及 `reused`/`pending`；仅有 8 个待处理分片时最多启动 8 个有效 worker。少于四个分片或最后收尾阶段允许部分 GPU 空闲，未引入重切分片或跨 split 混写。

## 2026-09-07 提交资源组选项

以 `e9a378e30ab34476517eed59be56e09ad9875f87` 为增量基线。核对仓库已有 A800 SFT/CPT 资源配置后，UI14 submit 支持 `aiai_locate`（默认，group_id=2146，专用队列）和 `yg`/`default`（group_id=1602，默认队列）。两者 cluster_id 均为 24，挂载目录和四卡 A800 型号相同。未添加 H20 或其他机器的实验 profile。

修改 `shell/ui14_cpt9000_a800.sh`、`scripts/submit_locany_ui5.py`、`scripts/locany_ui5_common.py`、`scripts/ui14_profile.py`、`scripts/ui14_checks.py`，将资源选项传递到提交、运行配置、YAML 和启动校验；新增 `tests/test_ui14_submission_resources.py`。文档补充分资源命令，参考 YG YAML 为 `jobs/rendered/locany_m32_cpt9000_ui14_a800x4_yg.yaml`。

本次 **70 项 CPU 回归全部通过，37.237 秒**；5 个 Python 文件 AST、实际 Bash 参数转发、`git diff --check` 通过。原 AIAI 参考 YAML 与当前渲染文本一致，新增 YG YAML 经 YAML 解析及正式配置校验通过。

```bash
python -m unittest tests.test_ui14_submission_resources tests.test_ui5_pipeline \
  tests.test_ui14_inference_workers tests.test_ui14_pipeline
```

- UI14 无参数默认 AIAI；旧 UI5 无 profile 仍默认 default；YG 别名恢复为 default。命令行覆盖 UI14_RESOURCE_GROUP 环境变量。
- 两套 YAML 的 cluster/group/queue/env 均与配置一致；两套 runtime 除 RESOURCE_* 字段外完全一致，启动重新解析后资源组、9000 CPT、4 卡、16k、每卡两个评测 worker、EVAL_FAIL_POLICY=stop 等约束保持。
- 未知资源、错误 groupIds、H20 机器、缺少参数或拼错 shell 参数会报错，不自动回退资源组。
- 使用构造 CPU ready 报告，分别执行完整提交入口并替换 mlx 为测试替身；先校验报告，再生成独立提交 YAML/runtime/binding，然后调用既有 `mlx job submitv2`。原 formal_job.yaml、formal_runtime.json 和 CPU 报告的字节及 mtime 不变。
- 报告校验失败不会发布提交 YAML，也不会调用 mlx；render-only 可以只渲染，但 binding 明确标注未提交。尝试以新资源 YAML 覆盖报告已绑定的文件会被拒绝。

本次仅本地 CPU 验证及参考 YAML 渲染，未读取真实集群数据，未调用真实 mlx 或提交训练。正式 submit 输出改为 `${UI14_DATA_ROOT}/submissions/formal_aiai_locate.yaml` 或 `formal_default.yaml`，旁边的 `.runtime.json`、`.binding.json` 记录资源、CPU 报告/产物摘要、repair_run_id 和 normalization_id。finalize 原产物继续作为数据验证依据，不因提交资源切换而重建。

## 2026-09-07 submit 进度与已运行进程观察

增量基线 `1c2903273dd498e5ec88c3f48e8a898fb8427e60`。提交入口在检查之前即输出 PID 和当前阶段；每 10 秒更新验证记录读取、产物摘要、修复绑定、图片清单与逐图检查进度，显示当前阶段 ETA。mlx 等待有心跳，服务端 ETA 标为不可估算。新协调进程使用独立提交锁，阻止并发重复运行；不会自动重试外部提交请求。

图片证据按唯一路径去重，一张图的 RGB/file hash 证据共享一次 stat；默认 16 线程、有界队列。变化文件的稳定内容摘要逐项继续写入既有 verification journal；中断后复用。未变图片不解码或重算 hash。原始证据仍约束内容，不能用新计算的错误摘要绕过原报告。旧代码已完成的 stat 没有续跑游标，重新启动仍需查询属性；本次通过并发降低该开销，不声称可以永久跳过变更检查。

新增只读 `submit-status --watch [--pid PID]`：同一 Linux 主机/用户可观察旧版正在运行的进程，读取 `/proc` 中的 PID 身份、打开文件的读指针及 mlx 子进程，不重新提交、重跑数据或终止原进程。旧文件字节进度受缓冲影响，ETA 只指当前文件；读取到 EOF 不算整个检查完成。新版优先显示进度 JSON；过期 PID 的日志不作为当前状态。原进程退出只报告进程退出，不据此声明任务提交成功。

**106 项 CPU 回归全部通过，125.952 秒。** 覆盖本次 10 项新回归及原有数据准备、并行裁图、验证重试、进度、正式资源和推理 worker 测试。Python AST 与 Bash 语法检查通过，实际 Bash 参数转发用 echo 替身验证，未调用 mlx。

```bash
python -m unittest tests.test_ui14_submit_progress tests.test_ui14_verification_retry \
  tests.test_ui14_submission_resources tests.test_ui14_progress tests.test_ui14_parallel_crops \
  tests.test_ui14_pipeline tests.test_ui5_pipeline tests.test_ui14_inference_workers
```

- 构造 4 张图、每图 2 种身份：并发屏障证明至少 2 个检查 worker 同时执行，每个路径只 stat 一次，Image.open 和图片 hash 函数禁止调用仍通过。
- 构造 3 张图仅改属性：先完成 1 张后中断，新检查复用该图内容结果并刷新另外 2 张；再跑 3 张全部复用。随后改 1 张像素，首次及再次检查均拒绝，不接受不匹配的缓存摘要。
- 完整提交函数以 mlx 测试替身验证：CPU 检查开始前已有日志；进入 mlx 时可见独立阶段；返回非零记录 failed 且仅调用一次；并发第二入口不能覆盖状态或发起提交。
- Linux `/proc` 使用本地构造目录验证进程筛选、读指针/字节 ETA、忽略追加 journal 的 EOF、mlx 子进程、退出和旧 PID 日志；本地 Windows 未运行真实 Linux 观察器或读取用户开发机进程。

本次实际仅执行本地 CPU 验证，未读取修复批次真实数据、未生成真实缓存、未测量远端检查耗时、未提交正式训练。原规范化 ID、数据/模型配置、两套资源 YAML 内容及 EVAL_FAIL_POLICY=stop 不变。当前旧进程的真实进度由用户在原开发机运行观察命令获取；不可将构造复用数量当作远端复用数量。

## 2026-09-08 正式任务首批超时与采样性能修复

读取用户上传的 trial 411146773 日志，确认任务已经进入 AIAI_locate 四卡 A800，训练在第一个 optimizer step 完成前因 rank 3 ALLREDUCE 等待 600 秒超时退出。数据构造耗时 3975.63 秒，其中 synth_occlusion 审核标注读入至采样初始化完成间隔约 51 分 34 秒。

核对并修复 `ui_defect_data.py` 每条抽样重新排序/洗牌全部源图的重复计算：按 task/polarity 和 source_cycle 复用一次排列，只保存当前 cycle；crop 轮换、seed/epoch/worker 序列、正负比例和人工样本顺序保持。训练入口增加各 rank/worker 的索引生成、首批 forward/backward 前后日志，不更改 collective 或超时设置。

**85 项 CPU 回归通过，102.847 秒。** 新增 5 项采样回归独立复现旧算法，覆盖 14 任务、多 seed/epoch、单侧来源、跨周期、恢复偏移、worker 隔离和 manual crop。构造 6,000 次抽样从 6.600 秒降至 0.300 秒，逐条相等；按日志中 170,847 条记录及正负源图规模构造，159,141 个索引生成耗时 8.394 秒，plan 构建另耗时 1.774 秒。均为本地 CPU 构造测量，不是 A800 实测。

日志缺少其余 rank/数据 worker 的超时调用栈，不能断言已排除全部 NCCL 原因。未执行修复后的四卡 backward，也未由本地重新提交；保留现有数据/cache/recipe，正常入口再次 submit 即可，正式参数不变。完整证据、变更与命令见 [trial 411146773 分析](ui14_training_failure_411146773.md)。

## 2026-09-08 启动建表与 step 0 全量评测

针对本次上传的 trial 411146773 日志：该次运行在第一个 optimizer step 完成前退出，尚未触发 100 步训练窗口；旧 UI14 profile 为 `EVAL_AT_START=0`，所以未产生初始评测。旧 Excel logger 在首次指标写入时才保存文件，诊断目录中缺少该文件与这次日志一致，不能据此判断当前远端任务的实时状态。

本次增量：UI14 正式 profile 及两份参考 YAML 固定 `EVAL_AT_START=1`，CPT checkpoint-9000 导出完整 14 任务 checkpoint-0 后先做全量 test 评测，成功才进入 SFT。pipeline 启动、独立评测入口以及 rank 0 Trainer 初始化均可创建或复用两个 sheet，打印 Excel 路径；不伪造 step 0 训练记录。100 步训练窗口、每 1000 步全量 14 项评测、每 GPU 两个推理 worker、16k SFT 和 `EVAL_FAIL_POLICY=stop` 保持。控制台复用旧五类 Image/BBox 与 five_task_macro 汇总，Excel 每轮仍为完整 36 行。

**本地实际执行 97 项 CPU 回归，38.590 秒，全部通过；9 个 Python 文件 AST、3 个 Shell 入口 `bash -n` 通过。** 两份跟踪的参考 YAML 与当前正式提交渲染函数逐字一致，AIAI_locate/2146 与 YG/1602 均核对 EVAL_AT_START=1、4 卡、2 workers/GPU、stop。

- 真正执行 Excel 写入/重读：首次启动只写两个表头；step 0 写完整 36 行；step 100 写训练窗口；重启不改既有字节和 mtime；旧表头迁移保留历史训练和评测内容。
- step 0 和 step 1000 评测集成：全部 14 项指标及来源/输入策略、CPT9000/SFT step 元数据保存；UI5 best 仍按原五类 macro；完整结果重跑不调用推理；删除 UI9 状态项或 Excel 单行均触发补齐，历史不重复。GPU 推理和 UI5 外部评分命令使用测试替身，UI9 评分、原图 gate 读取、Excel、history/best/state 实际执行。
- 执行真实 Bash 的 step 0 导出和评测控制块：仅模型导出/推理/完成状态命令替换为 CPU 测试函数；验证先导出再评测再放行训练、完整评测跳过，以及推理退出码 17 在 stop 策略下直接退出、不会进入训练。
- 提交入口复用绑定旧 EVAL_AT_START=0 的 finalize 配置快照和报告，在独立 submissions YAML 中生成新 EVAL_AT_START=1，原审计文件字节和 mtime 不变；真实 mlx 调用由测试替身代替，不需要重新生成数据/cache。

复现命令（已有 Pillow、NumPy、SciPy、PyYAML、openpyxl 的 CPU Python 环境）：

```bash
python -m unittest \
  tests.test_ui14_initial_evaluation tests.test_ui14_pipeline \
  tests.test_ui14_submission_resources tests.test_ui14_submit_progress \
  tests.test_ui14_inference_workers tests.test_ui5_excel_logger \
  tests.test_ui5_pipeline tests.test_ui_sampler_cycle_cache
```

本地未访问集群实时诊断目录，未加载 3B checkpoint、未执行 GPU step 0 推理/训练，也未提交新任务。真实 Excel 位于 `${WORKSPACE}/gui_models/locany-m32-cpt9000-ui14-a800x4-repair-v2/diagnostics/ui5_training_evaluation.xlsx`，由更新后的正式任务生成；已有失败运行没有记录的训练窗口不能补造。

## 2026-09-08 checkpoint-1000 评测 OOM：每卡一个进程

用户上传日志确认：训练已完成第一个 1000-step 分段，`TRAIN_EXIT_CODE=0`、`TRAIN_STATUS=SUCCESS`；随后 checkpoint resume 校验 `valid=true`、errors/warnings 为空，四 rank 的 optimizer、RNG、dataloader 状态均存在。此次故障发生在评测 synth_loneword 第 65/767 张，位置是语言模型 `modeling_qwen2.py:1500` 的 `logits.float()`；需要分配 3.71 GiB，仅余 2.02 GiB，同卡两个进程分别占 23.28 和 14.26 GiB。worker 物理 GPU=3，报错中的 GPU 0 是该进程隔离后的逻辑卡。

正式 UI14 profile、运行配置固定检查、两套参考 YAML 及 UI14 评测默认值统一从 2 workers/GPU 改为 1；四卡共 4 个槽位仍处理完整 14 项。视觉 FlashAttention 2、语言模型 SDPA、训练 12800/7268 token 预算、16k SFT、100 步训练窗口、step 0/每 1000 步评测和 EVAL_FAIL_POLICY=stop 均保持。没有改模型、processor、数据/crop、生成参数或预测完成标记。

**本次本地实际执行 99 项 CPU 回归，45.105 秒，全部通过；8 个 Python 文件 AST 通过。** AIAI_locate/2146 与 YG/1602 两份参考 YAML 均与当前正式渲染器逐字一致，4 卡 × 1 eval worker，12800/7268 和 stop 已核对。

- CPU 子进程和同步屏障验证四张逻辑卡各 1 个进程，14 个任务各处理一次；通用调度器旧的 2 workers/GPU 能力仍有回归覆盖，但当前正式 UI14 profile 拒绝覆盖为 2。
- 实际抽取推理脚本的纯 CPU 参数/manifest/待处理清单函数执行：构造两个完成结果（包括已完成的非法输出）、一张仅保存 OOM error sidecar 的图片；并发 2→1 和 GPU 分配变化不改变预测身份，两个结果及 raw/gate 字节和 mtime 保留，只有 OOM 图待重试；补齐后全部复用。更换 checkpoint 或几何清单摘要仍拒绝混用。此测试不导入模型/CUDA，也不解码图片。
- UI14 step 0/1000 完整评测、缺 UI9 补齐、36 行 Excel、旧五类 best、失败停止、分段 checkpoint 恢复、提交/启动正式参数和旧准备报告复用相关回归通过。提交用旧 2-worker 配置快照建立审计绑定，新实际 YAML 单独渲染为 1-worker，旧 CPU 报告和配置快照不变。

复现命令（具备项目 CPU 测试依赖）：

```bash
python -m unittest \
  tests.test_ui14_inference_resume tests.test_ui14_inference_workers \
  tests.test_ui14_initial_evaluation tests.test_ui14_pipeline \
  tests.test_ui14_submission_resources tests.test_ui14_submit_progress \
  tests.test_ui5_excel_logger tests.test_ui5_pipeline tests.test_ui_sampler_cycle_cache
```

远端恢复：等当前失败评测任务退出后，在同一目录 `git pull --ff-only origin codex/m32-cpt9000-ui14-v1 && bash shell/ui14_cpt9000_a800.sh submit`。保留当前输出目录和 checkpoint-1000，不执行之前从头重跑使用的归档命令，不重做数据/缓存/finalize。审计代码/config 身份变化可能使历史轮次重新检查/汇总；同一预测身份下已经落盘的逐图结果不会因并发减少而重推。先补齐本轮评测，再恢复到 1000 后继续训练。

本地仅检查上传日志并执行 CPU 回归，未连接集群、未实际重试失败图片、未测量单进程峰值、未提交新任务。减少同卡进程消除了本次观察到的双模型竞争，但不能据 CPU 测试声称已验证所有图片均无 OOM。

## 2026-09-08 仅 synth_loneword 独占 GPU，其他任务恢复双进程

按用户进一步修订，替代上一节的全局单进程设置：正式 UI14 profile 和两套 YAML 恢复 `EVAL_INFERENCE_WORKERS_PER_GPU=2`，评测入口显式传入 `--exclusive-gpu-tasks synth_loneword`。此调度属性不进入数据任务表，不改变预测身份、标注、模型、processor 或生成参数。

共享队列现在在同一把 condition 锁内领取任务及预留 GPU。synth_loneword 优先领取一张空闲物理卡，一个子进程占用该卡的全部调度容量；其所在卡的另一槽位等待，其他卡仍各启动两个进程。子进程退出后释放容量并唤醒等待槽位；任务失败或启动异常也释放，并停止继续领取任务，主流程仍遵守 EVAL_FAIL_POLICY=stop。日志和状态保存 exclusive_gpu_tasks、exclusive_gpu、gpu_slots_reserved。

**本次实际执行 102 项 CPU 回归，45.953 秒，全部通过。** 新增 3 项调度回归验证：

- 控制子进程执行时序，确保首批恰为独占任务 1 个进程加另外三卡各 2 个，任何时刻 synth_loneword 都不与同卡其他任务重叠；让独占任务先结束而其他三卡保持忙碌，验证释放卡的两个槽位都能继续领取普通任务，14 项各执行一次。
- 独占任务失败时唤醒等待的同卡槽位，后者不再领取队列中的任务。
- 子进程启动抛出 OSError 时释放预留卡、记录失败、返回非零，调度器不挂起。

其余回归继续覆盖四卡普通双进程/单进程队列、已有逐图结果与 OOM 图片续跑、完整 14 项 step 0/1000 评测、36 行 Excel、旧五类 best、提交资源及旧准备报告复用。推理命令测试核对 2 workers/GPU 且 exclusive-gpu-tasks 仅 synth_loneword。8 个修改 Python 文件 AST 通过；两份正式 YAML 与渲染器一致，12800/7268、初始评测和 stop 均不变。

已完成的 step 0 逐图结果继续按原预测身份复用。代码/config 审计信息变化可能触发历史轮次重新检查/汇总，但不会仅因本次调度策略变化而重推匹配的已有图片；没有放宽模型、生成参数、裁剪清单或 14 项完成检查。本地未进行真实 GPU 推理、未提交集群任务。恢复命令与上一节一致，保留当前输出目录和 checkpoint-1000，不归档、不清空、不加 overwrite。

## 实际数据统计的产生位置

派生根目录：
`/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_data/ui14_cpt9000_repair_v2`。

| 远端阶段 | 将产生的实际证据 |
|---|---|
| normalize（CPU） | 当前 manifest、repair_summary、18 份 JSONL 和图片可读性；source_snapshot.json、九项规范化 train/test、normalization_stats.json、ui9_page_split.json、ui9_image_overlap.json、parser_compatibility_issues.jsonl；cpu_check_report.json 的 normalization_complete，ready=false |
| cache-prepare（CPU） | 默认 16 线程图片指纹/尺寸扫描、去重、稳定分片；image_info.jsonl 续跑日志、summary.json 计数和各 split 的 ui14_prepare_ready.json |
| cache（GPU） | 仅新增七项 crop 任务的 train/test PP-OCRv5/OmniParser 检测，复用完成分片；不运行 prepare、merge、crop 或标签生成 |
| cache-finalize（CPU） | 合并检测、横向计划、派生标签、crop PNG、ui14_label_cache_ready.json、crop_index/images.jsonl、ui14_crop_complete.json、crop_performance 报告。周期评测不运行 detector |
| finalize（CPU） | 复用旧 UI5 审核 recipe/test cache；生成 training_recipe.json、evaluation_manifest.json、14 项连接检查、sampling_stats、完整 image_overlap、formal_job.yaml/formal_runtime.json；全部通过后 cpu_check_report.ready=true |
| submit | 按所选资源生成 submissions/formal_aiai_locate.yaml 或 formal_default.yaml 及 runtime/binding，摘要和资源检查通过后执行既有 mlx job submitv2。任务 ID 以远端 mlx 返回为准，本地未提交 |

报告的 post_repair_sources（normalize 时为 tasks）含每份文件的实际记录数、正负数、格式数量、GT 字段及修复数量。parser_comparison 分开给出 legacy_parse_failure_records、legacy_consumer_failure_records、parse_result_difference_records；页面统计覆盖 UI9 跨来源的 train/test 归属。完整字段解释和分阶段命令见 [运行文档](ui14_cpt9000_a800.md)。

参考 YAML 已提交为 [locany_m32_cpt9000_ui14_a800x4.yaml](../jobs/rendered/locany_m32_cpt9000_ui14_a800x4.yaml)。远端 finalize 会用同一渲染函数在派生根目录产生本次实际提交 YAML，并将其摘要绑定到 CPU 报告。

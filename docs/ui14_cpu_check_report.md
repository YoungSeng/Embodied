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

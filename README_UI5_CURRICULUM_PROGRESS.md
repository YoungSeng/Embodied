# UI5 curriculum 构建进度与耗时

课程数据构建器会在每个阶段开始、完成或失败时立即打印状态，并默认每 10 秒刷新一次：

```text
[CURRICULUM PROGRESS] ... stage="materialize_crop_pngs" completed=420/1000 crops percent=42.0% elapsed=00:03:30 build_elapsed=00:05:10 speed=2.00 crops/s stage_eta=00:04:50 last_progress_age=00:00:00 detail="..."
```

上面是格式示例，不是 H20 实测用时。

- `stage`：正在校验 bundle、检查图片/裁剪几何、匹配 anchor、生成 crop PNG，还是校验发布文件。
- `completed/total`、`percent`：当前阶段已经完成的数量与比例。
- `elapsed`、`build_elapsed`：当前阶段耗时、整个构建进程的耗时。
- `speed`、`stage_eta`：按当前阶段已完成工作估算的速度与剩余时间。初期样本不足或总量未知时显示 `unknown`。
- `last_progress_age`：距离最近一次完成工作过去多久。心跳持续但此数值增长，说明进程仍在当前图片或文件上等待/计算，不代表计数已经前进。
- `detail`：当前处理的图片、文件或样本，便于定位慢 I/O。

`stage_eta` 只表示当前阶段剩余时间。各阶段的单位和成本不同，它不是距离首次占用 GPU 或整个 1200-step 任务结束的时间。

进度信息同时保存在课程数据目录中：

```text
progress/build_progress.json       当前状态（原子替换）
progress/build_progress.jsonl      各次构建/校验的事件记录
```

这些文件是运行日志，不进入课程 manifest、数据身份或 `_SUCCESS.json` 的文件清单。正式 launcher 将控制台输出也保存到 `$OUTPUT_DIR/logs/curriculum-*.log`。preflight 结束后会清理临时数据目录，因此要保留其进度记录可使用已有的 `--keep-work-dir` 参数。

## CPU 先构建，H20 复用

对于会因 GPU 长时间空闲而回收的作业，先在可访问相同挂载盘的 CPU 作业中生成永久课程数据：

```bash
cd /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Embodied-ui5-curriculum

export WORKSPACE=/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace
export PYTHON_BIN=$WORKSPACE/conda_envs/LocateAnything/bin/python
export FROZEN_SELECTION=$WORKSPACE/gui_rollouts/ui5-train-rollout8-h20x2-v6-20260904/frozen/hour_009_20260904T180741Z-curriculum-v1
export ROLLOUT_BUNDLE_ROOT=$WORKSPACE/gui_data/ui5_train_rollout_bundle_v1
export CURRICULUM_DATA_DIR=$WORKSPACE/gui_data/ui5_curriculum/hour009-s42-v1

CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -u scripts/build_ui5_curriculum_recipe.py \
  --rollout-difficulty "$FROZEN_SELECTION/complete8.jsonl" \
  --rollout-bundle-root "$ROLLOUT_BUNDLE_ROOT" \
  --output-dir "$CURRICULUM_DATA_DIR" \
  --seed 42 \
  --progress-interval-seconds 10
```

复用目录内完整且身份一致的成品时，构建器仍进行完整性校验，这些校验也会显示进度。数据目录中的图片使用绝对路径，所以生成后保留这个目录的位置。

在之前已经配置好 hour009、评测清单、新 `RUN_NAME/OUTPUT_DIR` 的 H20 作业中，继续指定相同课程数据目录即可：

```bash
export CURRICULUM_DATA_DIR=/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_data/ui5_curriculum/hour009-s42-v1
export CURRICULUM_PROGRESS_INTERVAL_SECONDS=10

CUDA_VISIBLE_DEVICES=0,1 \
bash shell/run_locany_ui5_crop_rollout4_curriculum_h20x2.sh
```

launcher 和 preflight 都支持 `CURRICULUM_PROGRESS_INTERVAL_SECONDS`；直接运行 Python 构建器时使用 `--progress-interval-seconds`。只调整此间隔不会改变采样、crop 内容、课程 manifest 或训练设置。

已经运行的旧 Python 进程不会自动获得新日志。更新代码后，需要重新启动构建器才能使用进度功能；不要同时启动两个构建器写同一个课程目录。此进度功能以及单独构建课程数据均不需要 GPU。

在其他终端查看永久目录的进度：

```bash
tail -F /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_data/ui5_curriculum/hour009-s42-v1/progress/build_progress.jsonl
```

独立 preflight 仍会构建临时课程并执行原有检查。CPU 永久构建完成后，可直接交给正式 launcher 校验和复用，无须为复用目的再运行一次独立 preflight。

## 切换快照并复用全部 PNG（hour009 → hour018）

新增 `--reuse-crops-from`，从已经完整发布的 v4 课程目录导入图片资产。
新快照仍须重新冻结、从新 summary 读取 hard ID/数量、匹配 anchor、重建三池索引。
不会复用旧快照的标签分组、训练状态、评测结果或 best checkpoint 记录。

复用严格要求同一个不可变 bundle、完整相同的 crop ID 集，以及逐项相同的源图
SHA256、sample ID、crop 坐标和尺寸。旧 manifest、`crop_assets.jsonl`、成功标记、
PNG 大小/哈希均参与验证。图片使用硬链接，源文件不移动、不重新裁剪、不重新编码。
因此源/目标必须在支持硬链接的同一个文件系统；不支持、缺文件、损坏、身份不符时直接
失败，没有复制或重新裁图的自动降级。源数据已经全量覆盖 bundle，所以同一 bundle 的
后续快照可以复用所有 crop，不受 hard/anchor/global replay 归属变化影响。

完成旧构建后，构建器可单独运行：

```bash
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -u scripts/build_ui5_curriculum_recipe.py \
  --rollout-difficulty "$NEW_FROZEN_SELECTION/complete8.jsonl" \
  --rollout-bundle-root "$ROLLOUT_BUNDLE_ROOT" \
  --output-dir "$NEW_CURRICULUM_DATA_DIR" \
  --reuse-crops-from "$WORKSPACE/gui_data/ui5_curriculum/hour009-s42-v1" \
  --seed 42 --progress-interval-seconds 10
```

控制台会显示 `stage="reuse_crop_pngs"` 和 `[CROP ASSETS] total=... reused=... generated=0`。
课程 manifest 中的 `crop_asset_reuse` 记录复用来源、来源身份和复用数量，并参与新课程身份
计算。正式 launcher / preflight 对应环境变量为 `CURRICULUM_REUSE_CROPS_FROM`。
完整目标目录再次启动时走原有完整性校验，不重复链接。preflight 在复用模式下默认把临时
目录放在源课程的父目录，以保持同一个文件系统；不要显式指定另一文件系统的临时目录。

### 当前旧前台流程仍在裁图：在同一开发机另开一个终端执行

新入口在前台等待，不使用 nohup 或后台启动。`--take-over-builder-pid` 指定的是
旧裁图子进程，而不是父进程：它核对命令行、输出目录、父子关系和内核进程身份后，
只对旧的 `python -u -` 自动提交包装器发送 SIGSTOP。裁图子进程保持运行，旧窗口应保持连接。
待源课程完整发布且裁图子进程退出，再终止旧包装器，防止其继续提交 hour009。
随后冻结指定快照、CPU 构建新的复用课程、提交 H20×2。代码不自动停止已提交到平台的作业。

```bash
cd /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/code/Embodied-ui5-curriculum
git pull --ff-only origin codex/ui5-crop-rollout4-curriculum-hard114

/mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/conda_envs/LocateAnything/bin/python -u \
  scripts/prepare_ui5_curriculum_snapshot.py \
  --snapshot /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_rollouts/ui5-train-rollout8-h20x2-v6-20260904/snapshots/hour_018_20260905T030758Z \
  --reuse-crops-from /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_data/ui5_curriculum/hour009-s42-v1 \
  --previous-submission-dir /mnt/bn/intelligent-service-arnold-hl/logging/sicheng_workspace/gui_logs/ui5_curriculum/locany-ui5-crop-rollout4-curriculum-hour009-h20x2-sdpa7268-20260904T204242Z-276590 \
  --take-over-builder-pid 1883296
```

`1883296` 是本次日志中的旧构建 PID，不是长期固定值。如果 PID/父进程身份不符，脚本拒绝
发信号，不会猜测要停止哪个进程。需要 Linux/Python 的 pidfd 支持。若源目录已完整发布，
且没有正在运行的旧构建/自动提交器，可以不传 `--take-over-builder-pid`。

每次切换使用新的 frozen 目录、课程目录、RUN_NAME、OUTPUT_DIR 和评测身份。
资源沿用现有 H20×2 的 Arnold 镜像/队列/挂载；GPU 作业开始后先校验完整课程，再进入
step 0 baseline 和 200-step 训练/评测循环。没有指标早停或 Magi 降级。

提交状态写入打印出的 `snapshot-switch.json`。旧流程已尝试提交、任意校验失败或出现
复用之外的新生成 PNG，都会阻止提交；提交失败也不自动重试，避免不确定状态下重复申请。
切换失败/中断时，已经暂停的旧自动提交器保持暂停，以免意外提交 hour009；状态文件记录
其 PID 和启动身份。不要盲目 SIGCONT 或重复启动，请先检查源构建与平台任务状态。

跨快照复用省掉的是重新裁图/编码/写入图片内容，仍有哈希校验、链接和索引生成。
不要把当前阶段 ETA 理解为整个切换或训练完成 ETA，也不保证获配 GPU 后的校验一定只需几分钟。

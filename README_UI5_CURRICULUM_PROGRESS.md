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

# UI14 trial 411146773 日志分析与采样修复

输入为用户提供的 `sicheng.yang-424de255e5789699-411146773-1788800269230.zip`，SHA256：`5a228fe47176ac140b6ae581adc2739af30ee01beeb5deda5d487e59b6a2a508`。只读取其中三个日志文件，未连接集群。下列行号指压缩包内 worker-0 的 stdout，以换行/回车分行；平台 stderr 末尾的 exit code 1 是训练失败的汇总。

## 日志确认的结果

- 正式任务已经进入 AIAI_locate，group_id=2146，A800-SXM-40GB × 4。
- 22:52:52 用户脚本开始；00:06:30 才进入 Trainer 训练循环。训练数据构造自身记录耗时 `3975.63s`（约 66 分 16 秒，stdout 3000）。
- 最大来源 synth_occlusion 在 23:12:37 报告已读入审核标注，至 00:04:11 才完成采样初始化：这段间隔约 51 分 34 秒（stdout 2941–2942）。原 JSONL 建索引仅记录 0.36 秒，不能把这 51 分钟全部归为图片读取。
- 00:13:08 仅看到 rank 3 的首批 forward 审核记录；00:23:08 rank 3 的 `ALLREDUCE` 超时，`SeqNum=1461`，`NumelIn=78839327`，超时限制 600000ms（stdout 5238–5239）。它在 backward 中等待，随后 SIGABRT，torchrun 退出 1。
- 进度最后仍为 `0/16000`。日志没有成功完成 SFT optimizer step 或保存本轮 SFT checkpoint 的证据；首个 0→1000 训练分段失败，尚未进入 1k 全量评测。
- `libGL.so.1` 预检失败后脚本安装了运行依赖，并继续到模型训练；它不是此后 NCCL 超时的直接报错。日志没有 CUDA OOM；所附监控采样的最大显存为 25,358 MiB，不能据此将同步问题归因于显存不足。

## 确定的代码缺陷与证据边界

基线 `a6cbed03365c90bb293dfbbd8a750e21adba61ec` 的 `_rotating_source_record` 每取一条记录，都重新 `sorted(source_groups)` 并用相同种子把全部源图 shuffle 一遍。同一轮换周期里，绝大多数调用生成的是同一排列。`materialize_task_source_balanced_rotating_indices` 对整轮抽样重复调用它，源图排序/洗牌开销随源图数量和抽样数的乘积增长。

synth_occlusion 的日志规模为：正源图 53,047、负源图 41,118，正记录 80,823、负记录 90,024，共 170,847 条 crop 记录；一轮抽样 159,141 次。仅源图 shuffle 就需要反复处理约 71.8 亿个源图元素，还不含排序开销。

同样的 materialize 路径既在 dataset 初始化的首轮校验执行，也在每个 DataLoader worker 首次访问某个来源/epoch 时执行。各 worker 的来源种子和首次访问时间不同；这提供了首批取数速度悬殊的明确代码解释。

日志没有 rank 0/1/2 或各 DataLoader worker 在超时瞬间的完整调用栈，因此无法证明它是本次 NCCL 超时的唯一原因。当前修复移除这个已确认的瓶颈，不扩大 NCCL timeout、不绕过梯度同步、不更改 M32 分支或损失；四卡正式复跑仍是最终验证。

## 修复范围

- `eaglevl/train/ui_defect_data.py`：每个 task/polarity 的一次 materialize 调用只排序一次源图；只在 source_cycle 改变时重算排列。缓存只保留当前周期，在调用结束时释放。seed、随机数命名空间、任务交织、正负比例、单侧来源、manual_gt_repair 优先和 crop 轮换规则不变。
- `eaglevl/train/locany_finetune_magi_stream.py`：打印各 rank 的初始采样校验、worker/PID/seed/epoch 的索引生成开始与结束、耗时和数量。各 rank 首个 microbatch 在 forward/backward 前后分别打印标记，原首批 forward 审核保留；不新增 collective。
- `tests/test_ui_sampler_cycle_cache.py`：独立保留旧采样算法作为 CPU 对照，逐条比较 14 任务、多种 seed/epoch、单任务/单侧来源、跨周期、人工 crop、worker 隔离及恢复偏移。

## 已执行的本地 CPU 测量

| 构造场景 | 旧实现 | 新实现 | 验证 |
|---|---:|---:|---|
| 正源图 2,000、负源图 1,501，6,000 次抽样 | 6.600s | 0.300s | 全序列一致；本地约 22 倍 |
| 与日志相同的源图数和记录数，159,141 次抽样 | 未运行全规模旧算法 | 8.394s | 构造 plan 另耗时 1.774s；索引完整 |

第二行的数据内容是构造的，仅规模与日志相同。结果不能作为真实挂载盘、A800 或完整训练吞吐。新大规模抽样序列 SHA256（JSON 紧凑序列化）为 `654840243bd6407ceb75e1602df289fa965969b44e47f484120933ebf11972f0`。

独立复杂度验证：6,000 次抽样的源图 shuffle 次数随实际跨越周期计数，不超过 5 次；测试不使用耗时阈值。逐条等价验证保证优化不改变抽样轨迹，而非只比较分布。

本次 **85 项 CPU 回归全部通过，102.847 秒**；3 个 Python 文件 AST 与 `git diff --check` 通过。GPU/真实 DeepSpeed backward 未执行。

```bash
python -m unittest tests.test_ui_sampler_cycle_cache tests.test_ui14_pipeline \
  tests.test_ui5_pipeline tests.test_ui14_inference_workers \
  tests.test_ui14_submission_resources tests.test_ui14_submit_progress
```

## 正式重提

这份日志对应的 trial 已 FAILED。保留当前数据目录、缓存和训练输出，不需要重新执行 normalize/cache-prepare/cache/cache-finalize/finalize。提交前仍按原入口验证已准备数据；本次仅采样运行代码和诊断日志改变，数据绑定和 YAML 参数不变。

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui14-cpt9000
git pull --ff-only origin codex/m32-cpt9000-ui14-v1 &&
bash shell/ui14_cpt9000_a800.sh submit
```

先确认没有另一份正在运行的重复任务。正式 profile 继续 CPT checkpoint-9000、四卡 A800、16k SFT、每 100 optimizer step 训练记录、每 1k 完整 14 项评测、每卡两个评测推理 worker、EVAL_FAIL_POLICY=stop。本次没有从日志发现可恢复的已训练 SFT step；若输出目录实际存在其他有效 SFT checkpoint，仍由原有恢复与批次绑定检查决定。

此报告及修复在本地完成；未由本地执行真实集群提交，也未验证修复后的四卡 backward。

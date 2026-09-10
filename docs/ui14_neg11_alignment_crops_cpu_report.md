# Alignment crops CPU 验证

父代码：4bf4ec646a46b22fa7dffe1affbf8e4e50cb9433。
本次按最终更正实施：content_missing 全图；ui_alignment crops 训练和推理。
使用独立分支、checkout、数据目录和训练输出，保持当前 neg11 available 选择与原图 test。

本地 CPU 新增 7 项测试通过（6 项集成回归 32.564 秒，1 项 CPU/GPU 交接回归 4.916 秒），覆盖：

- 两个换行任务各自占满物理卡的两个逻辑槽位，其余卡仍允许两进程，14 任务各执行一次。
- 正式渲染器及 YAML 校验器：4×A800、CPT-9000、16k、100/1000 steps、step 0、
  EVAL_FAIL_POLICY=stop、legacy 解码、独立 OUTPUT_DIR、两个独占卡任务。
- 真实构造图片建立父 neg11 available 数据；保留实际缺口，不恢复严格配额。
  新 prepare 只读复用父版本（禁止 Image.open 时仍成功），ready=false 等待缓存。
  用固定原始 detector 输出运行真实 GT-free 几何、PNG、标签、覆盖及完成标记流程。
  对齐训练改为 crop，测试保留原图清单；content_missing 全图；其他 13 项 recipe 原样保留。
- 新增 alignment crop 完成后，真实 finalize/cache 校验成功；再次 finalize 和 prepare
  不解码图片，PNG 字节和文件属性不变；父版本所有文件字节/属性不变。
  新 checkpoint registry 保存 crops 策略；旧 full_image registry 仍兼容，其他任务不能任意改视图。
- 缓存任务选择只产生 ui_alignment/train 与 ui_alignment/test 两个作业。
  推理 worker 从 saved view_policy 生成 detector_scan 参数，content_missing 参数为 full_image。
- 实际 CPU cache-prepare 处理复制自父目录的输入指针，保留冻结文件字节，独立生成新目录指针。
  再次准备不解码图片；模拟 GPU 调用只派发 text/text/icon/icon，禁止图片扫描及隐式准备。
  任一 split 缺少本地交接清单时，在启动 GPU worker 前失败。
- 独立旧模型补评显式 legacy，覆盖残留结构约束环境变量；两种策略比较身份不同。
  快照文件继续复用，旧权重不修改，旧结构约束预测不复用。

既有 17 项导入、断点续跑、单/双 worker、loneword 独占、失败调度、父数据只读冻结、
快照中断恢复与正式 profile 回归也通过。测试使用 CPU 子进程和小型图片；
继承的 UI5 测试缓存仅为构造标记，其验证在集成 fixture 中替换，
实际 UI9 和本次新增 alignment 的 cache 检查使用真实实现。

```bash
python -m unittest tests.test_ui14_alignment_crops \
  tests.test_ui14_inference_imports tests.test_ui14_inference_resume \
  tests.test_ui14_inference_workers tests.test_ui14_alignment_context \
  tests.test_ui14_cache_stages.StageSeparationTests.test_frozen_alignment_local_handoff_prepares_only_two_splits_and_gpu_never_scans -v
```

参考正式 YAML 已由生产渲染器生成并校验；shell 入口通过 bash -n。
模型结构、损失、375 坐标投影、评分代码及非法输出惩罚未修改。
本次扩展任务注册表验证仅允许 ui_alignment 显式选择 full_image/crops，随 checkpoint 保存。

本环境不能访问 A800：尚未执行实际数据 prepare/cache-prepare、GPU detector、
真实 cache-finalize、GPU 补评或正式提交。真实原图/crop 数、复用数量、吞吐和 ETA
由集群对应阶段的日志、progress.json、alignment_crop_check.json 与 CPU 报告提供。
未停止旧 neg11 训练，未修改其 checkout、数据、checkpoint、Excel 或状态。

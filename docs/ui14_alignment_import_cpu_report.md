# Alignment worker 导入修复（2026-09-10）

基线：4d5e314fcea1a9c49d9374f7b9a1999763a11be9。
用户实际 traceback 在导入 eaglevl.ui_task_registry 时失败，尚未执行模型加载。
本地检查该文件已跟踪且存在；旧推理入口直到导入 ui14_common 才添加项目根目录，
此时 eaglevl 已经可能从旧 editable checkout 导入并缓存。

修复将根路径设置和包来源核对提前到 torch/transformers/eaglevl 子模块导入之前，
仍保留原 CUDA mask/re-exec 顺序。启动打印实际 checkout/package 路径；
本 checkout 缺文件或进程预加载另一 checkout 时明确报错，不替换已加载模块。
不改变生成语法、模型结构、数据、评分、正式参数或调度。

实际执行本地 CPU 回归 13 项，全部通过，2.355 秒：

- 用独立进程执行真实推理脚本；PYTHONPATH 指向缺少任务注册表的旧包，cwd 为无关目录。
  在第一处 torch 导入使用 CPU 探针停止，确认实际 registry 来自当前 checkout，ID 为 0–13。
- 同一 fixture 禁用提前设置路径，复现用户的同名 ModuleNotFoundError。
- 缺失 registry 与 sitecustomize 预加载旧包两种情况均在模型依赖之前报告明确错误。
- CUDA mask 隔离后的替代进程仍加载当前包。Windows 上校验原 execvpe 的 argv/env，
  再显式启动替代进程；没有把 Windows 运行称作 A800/Linux 实测。
- 原结果续跑不重写已完成输出；并发数变化不改变预测身份，运行失败样本仍为待处理。
- 4 卡 8 槽位、14 任务队列、loneword 独占卡及失败后的调度释放行为回归通过。
- 原正式 YAML/运行配置的每卡两个评测槽位校验通过。

```bash
python -m unittest tests.test_ui14_inference_imports \
  tests.test_ui14_inference_resume tests.test_ui14_inference_workers -v
```

本次仅修改推理入口、新增 CPU 回归及文档。原 decoder_contract 已包含推理文件摘要，
因此更新后运行 CPU eval-prepare 刷新比较身份，随后 eval-existing。
沿用既有快照逐文件复用；不需要重新 audit、prepare、normalize 或 detector。

本地未连接 A800；未执行真实 eval-prepare、GPU 补评或集群提交。
未修改旧训练 checkout、数据、checkpoint、Excel 或运行状态。

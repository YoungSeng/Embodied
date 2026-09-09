# Alignment 路径接续修复（2026-09-10）

基线为 alignment 分支 533f36b。本次只修改入口路径选择、写入前检查和已有审计导入，
没有修改生成/评分代码、训练配置、已冻结数据或参考 YAML。

用户提供的日志确认 step-4000 审计完成：1956 images、missing_raw=0、status=complete，
但输出位于旧 ui14_cpt9000_neg11_v1/historical_audit。
随后 prepare 在读取父目录 negative_extension_manifest.json 时失败；
所贴 traceback 未包含最终缺失文件的完整路径，因此无法由日志直接确认实际 parent_root。
源码的两个通用 UI14 环境变量默认值会继承旧终端变量，足以重现输出/父目录错位。
且原保护仅比较所传 parent_root，传入旧 repair 父目录时未能拦住向旧 neg11 目录写日志。

本次更正：

- Bash 与直接 Python 入口均使用 UI14_ALIGNMENT_DATA_ROOT / UI14_ALIGNMENT_PARENT_DATA_ROOT。
  通用旧变量不作为默认路径。启动打印实际输出、父数据、历史 run/test 和新训练目录。
- 写日志、锁和验证 journal 前保护已知旧目录与没有 frozen_parent 标记的既有 UI14 数据目录。
  prepare 父目录缺文件时显示实际父路径、缺失文件和正确配置项；不会新建输出目录。
- 新增 --reuse-audit-root。校验旧完成状态、审计身份、task/step 和四个产物摘要，
  原子复制文件后才发布新目录 summary。源文件只读、不建立硬链接；可重复接续。
  原 audit-errors 随后按完整输入/代码身份决定复用或重建，缺项/变更不跳过。
- 原 ui14_alignment_audit.py、ui_answer_grammar.py、评分脚本字节未修改，
  因而本次入口修复不会单独使旧审计身份失效。

实际在本地 CPU 执行 11 项测试全部通过（2.422 秒）：

- 通用残留变量隔离；专属变量与 CLI 优先级；实际 Bash wrapper 的变量传递。
- 即便指定错误 parent，仍阻止向已知旧目录写入；自定义既有数据目录同样受保护。
- 缺失扩展清单时，在新输出目录建立前失败，错误包含父路径和配置项。
- 残留变量场景下，audit 的日志/状态只写新目录，旧数据文件字节和属性不变。
- 完成审计跨目录导入后，替换评分函数为抛错仍能复用两 step；
  重复导入 reused=2，源文件字节/mtime/ctime 均保持一致。
- 原 raw 改变时生成新审计身份；损坏/半成品审计不能发布完成导入。
- 原审计逐图计数和基线评分函数一致性回归通过。

测试使用小型构造图片与预测，不是集群真实审计。未连接 A800、未移动用户实际文件、
未执行真实 prepare/eval-prepare、GPU 补评或提交训练。

```bash
python -m unittest tests.test_ui14_alignment_paths tests.test_ui14_alignment_audit -v
```

完整续跑命令见 [README](ui14_alignment_context_README.md#旧环境变量导致串目录后的接续)。

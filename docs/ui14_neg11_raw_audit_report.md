# RawImgURL 待确认时的 CPU 审计增量

2026-09-09，基于 `3cddb1b2362cd2fe58024cd39fdcbb09b7137217`。用户已明确表示尚不能确认七个合成来源的 RawImgURL 是否都是当前任务无缺陷的合成前底图，因此本增量没有放宽 raw 准入、没有增加默认信任开关。

用户回传的 inventory 统计（本机未访问 A800）：

| 范围 | 独立正例目标 | 已选独立正常图 | 缺口 |
|---|---:|---:|---:|
| train | 86,130 | 11,124 | 75,006 |
| test | 9,600 | 1,199 | 8,401 |
| 合计 | 95,730 | 12,323 | 83,407 |

最大排除项是 99,368 条 raw 引用缺少来源证据。它们是引用条数，不能作为独立正常图数量抵扣缺口。还存在已观察到的跨 split 冲突及重复内容，不能用取消去重/取消 split 检查来补齐。

新增入口：

```bash
cd /mnt/bn/intelligent-service-yg/logging/sicheng_workspace/code/Eagle_LocateUI5_v4/Embodied-ui14-neg11
git pull --ff-only
bash shell/ui14_neg11_a800.sh audit-raw
```

仅 CPU；输出 `/mnt/bn/intelligent-service-yg/logging/sicheng_workspace/gui_data/ui14_cpt9000_neg11_v1/raw_reference_audit/`。报告对比 raw 与其配对 ScreenShot/Local 的 RGB 身份，记录引用路径、源图/记录/页面关系和 raw 内容在合成任务中的 split 关联，并生成最多每任务 10 个配对样例。内容相等是可核对的关系，不自动把未确认 raw 转为负标签。

修改文件：`scripts/ui14_neg11_raw_audit.py`、`scripts/ui14_neg11.py`、README，以及 `tests/test_ui14_neg11_raw_audit.py`。其余训练、采样、缓存和评分规则不变。

实际 CPU 验证：

```bash
python -m unittest tests.test_ui14_neg11_raw_audit tests.test_ui14_progress
```

12 tests，5.841 秒，OK。构造用例覆盖同内容不同路径去重、缺失引用、raw=异常图、raw=Local、跨任务 train/test raw 冲突、已选负例识别、HTML GT 默认隐藏；重复审计禁止 Image.open 仍通过。逐文件核对原始图片、旧规范化数据、当前 inventory、selection、页面映射的字节和 mtime，均未修改。

本机仅执行上述 CPU 构造验证；尚未运行 A800 的真实 audit-raw，没有生产 raw 独立数量或来源语义结论，没有执行 GPU 或提交训练。

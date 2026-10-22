# 小组作业工作区总览（5 个地方，各建什么、谁建、怎么验收）

> 一个人（组长）今晚就能全部建完，其余 7 人 10 分钟内接入。
> 判定标准只有一条：**每个人都能独立完成"拉代码 / 拿数据 / 写文档"三件事，而不需要问别人要文件。**

---

## 工作站 1 · GitHub 私有仓库（代码真源）

| 项 | 值 |
|---|---|
| 仓库名 | `am-defect-tl`（本仓库） |
| 可见性 | **Private**（提交前可改 public，也可以一直 private 到最后打包 zip） |
| 建的人 | 组长 |
| 权限 | 8 人全部 **Write**；`main` 设 **分支保护**：Require PR + 1 approval + Require status checks (ci) |
| 建好标志 | 8 人都能 `git clone` 且 PR 模板能自动弹出 |

**从本机推到 GitHub（组长执行一次）**

```bash
cd am-defect-tl
git remote add origin https://github.com/<你的用户名>/am-defect-tl.git
git push -u origin main
```

然后在 GitHub 网页：Settings → Collaborators 加 7 人 → Settings → Branches → 保护 `main`。

---

## 工作站 2 · Kaggle 私有 Dataset（数据真源）

| 项 | 值 |
|---|---|
| Dataset 名 | `am-defect-v1`（slug 形如 `<用户名>/am-defect-v1`） |
| 可见性 | **Private** |
| 内容 | `raw/`（D1+D2 原始，896 MB）、`processed/`（224×224）、`splits/`、`manifest.csv` |
| 建的人 | 数据管理员（地基组） |
| 挂载路径 | `/kaggle/input/datasets/kennethdeng/am-defect-v1/data/` ← **注意新版多了一层** |
| 现状 | ✅ 2026-10-23 已上传并验证：854.6 MB，136/135/947/2070 全对 |
| 建好标志 | 8 人各开一个 Notebook，`os.listdir(BASE)` 能列出四个分组目录 |

```bash
# 本机（数据管理员）
pip install kaggle                       # 先在 kaggle.com/account 下载 kaggle.json 放到 ~/.kaggle/
kaggle datasets create -p am-defect-v1 -r zip
# 之后更新版本：kaggle datasets version -p am-defect-v1 -m "add splits"
```

> 数据**不进 Git**。每个组员不需要下载 896 MB，只需要挂载。

---

## 工作站 3 · Kaggle 账号池（算力）

| 项 | 值 |
|---|---|
| 数量 | 8 个（每人一个，**不要共用账号**） |
| 要求 | 完成手机验证才能拿到 GPU 额度（约 30 小时/周） |
| 建好标志 | 每人截图 `/kaggle/` 首页能看到 "GPU T4 x2 / P100" 选项 |
| 对接 | GitHub PAT 存进 Kaggle → Add-ons → Secrets，Notebook 里用来 clone 私有仓 |

---

## 工作站 4 · 飞书（沟通 + 看板，沿用选题阶段的环境）

| 项 | 值 |
|---|---|
| 群 | `IDAT7215 Topic4 小组`（8 人 + 置顶消息） |
| 多维表格 | `Topic4 任务看板`：字段 = 任务 / 组 / 负责人 / 截止 / 状态 / PR 链接 / run_id |
| 文档 | `Topic4 组会纪要`（每次会后 10 分钟内贴结论 + 待办） |
| 建好标志 | 8 条任务在建，每人至少 1 条；置顶消息里有 5 个工作站链接 |

---

## 工作站 5 · Word Online（报告定稿）

| 项 | 值 |
|---|---|
| 位置 | HKU M365 → Word → `IDAT7215_Topic4_Report.docx`（放组内 OneDrive/SharePoint 共享文件夹） |
| 权限 | 8 人可编辑 + 组长可管理 |
| 首页内容 | 组号、8 人姓名学号、**章节所有权 + 字数预算**（见 README 第 8 节） |
| 建好标志 | 8 人都能打开并看到自己的章节标题；字数统计 = 0 但有结构 |

> 为什么不用飞书文档做定稿：Moodle 只收 docx，飞书导出后图表编号/字数/字体容易乱。
> 飞书只放**大纲、讨论、纪要**，正文一律在 Word Online。

---

## 建好后的验收（5 分钟，组长逐条打勾）

- [ ] 8 人全部接受 GitHub 邀请，且能 `git clone`（截图发群）
- [ ] `main` 已开分支保护，直接 push 会被拒绝（自己试一次）
- [ ] PR 模板自动出现；CI 在第一个 PR 上跑绿
- [ ] Kaggle Dataset `am-defect-v1` 建好，8 人验证挂载路径可读
- [ ] 8 个 Kaggle 账号都通过手机验证，能看到 GPU 选项
- [ ] 飞书看板 8 条任务建好、认领完毕
- [ ] Word Online 报告文档建好，首页写了章节所有权
- [ ] 群里置顶一条消息，包含以上 5 个链接 + 第一次冒烟 deadline（11/1）

---

## 常见问题

**Q：GitHub 打不开 / clone 很慢怎么办？**
先试 GitHub Desktop（走系统代理更稳）；仍不行就在 Gitee 建同名私有镜像，每周同步一次，但 **GitHub 是唯一真源**。

**Q：有人完全不会 Git 怎么办？**
第一周用 GitHub Desktop 的图形界面（克隆/拉取/提交/推送/开 PR 全是按钮）；组长投屏带一次 30 分钟。
第二周起必须人人会用，否则最后一周合并会崩。

**Q：数据放在 Kaggle，那本地没法调试怎么办？**
本地只放 `processed/` 的 224 图（约 120 MB）就够跑通代码；896 MB 原图只有数据管理员需要。

**Q：报告能不能直接在飞书写？**
可以写草稿，但**定稿必须搬到 Word Online**，否则格式检查那一关会花掉整个周末。

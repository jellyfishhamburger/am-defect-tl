# 组长云端配置手册（GitHub / Kaggle / Word Online / 飞书）

> 目标：**8 个人都能"拉代码 / 拿数据 / 写文档"三件事**，不需要互相要文件。
> 本文按"谁做、怎么做、怎么验收"写，打勾即完成。

---

## 0. 当前状态（2026-10-23）

| # | 事项 | 状态 | 备注 |
|---|---|---|---|
| 1 | GitHub 私有仓 `am-defect-tl` | ✅ **已建好** | https://github.com/Zheyi-D/am-defect-tl |
| 2 | 仓库骨架（36 文件 / 2 commits） | ✅ **已推送** | README、契约、CI、测试、paper_results.csv |
| 3 | CI（编译 + 契约测试） | ✅ **已跑通** | 每次 push/PR 自动运行，10 秒 |
| 4 | 任务看板（7 个 Issue + 8 个标签） | ✅ **已建好** | Issues #1–#7，按组分好标签 |
| 5 | 加 7 个组员为 Collaborator | ⬜ **待办** | 需要收集 7 人的 GitHub 用户名 |
| 6 | `main` 分支保护 | ⚠️ **受限** | GitHub Free 私有仓不支持，见 §1.2 三个解法 |
| 7 | Kaggle 私有 Dataset `am-defect-v1` | ⬜ **待办** | 需要你的 Kaggle 账号，见 §2 |
| 8 | Word Online 报告文档 | ⬜ **待办** | 见 §3 |
| 9 | 飞书群 + 置顶 + 看板 | ⬜ **待办** | 见 §4 |

---

## 1. GitHub

### 1.1 加人（5 分钟，需要 7 个用户名）

网页：`Settings → Collaborators → Add people` → 输入对方 **GitHub 用户名**（不是昵称、不是邮箱，除非他设了公开邮箱）。

也可以用命令批量加（把 `u1 u2 ...` 换掉）：

```powershell
foreach ($u in @("u1","u2","u3","u4","u5","u6","u7")) {
  gh api -X PUT "repos/Zheyi-D/am-defect-tl/collaborators/$u" -f permission=push
}
```

> 邀请会发到对方邮箱，**7 天不点会过期**。过期后重新加一次即可。

### 1.2 分支保护的坑（重要）

GitHub **Free 计划不支持私有仓的分支保护 / rulesets**（会返回 403 `Upgrade to GitHub Pro`）。三个解法：

| 方案 | 优点 | 缺点 | 建议 |
|---|---|---|---|
| **A. 申请 GitHub Student Developer Pack** | 免费拿到 Pro，私有仓也能开保护 | 需要学生邮箱验证，可能要等 1–2 天 | ⭐ **推荐**，HKU 邮箱可申请 |
| B. 仓库改成 Public | 立刻能开保护 | 别的组能抄我们代码 | 只在提交后再考虑 |
| C. 纯约定（不做技术限制） | 零成本 | 有人直接推 `main` 时拦不住 | 兜底方案 |

选 C 的话，就把下面这条写进群公告：

> **不要直接 push `main`。** 所有人开分支 → 提 PR → 组长 merge。
> 技术上拦不住，但 GitHub 会记录是谁推的，作业里有据可查。

### 1.3 你怎么确认"保护/流程生效"

- 让一个组员故意直接在 `main` 上改文件并 push —— 若能推上去，说明保护没生效；
- 让一个组员随便改个 README 提 PR —— 应该看到：
  - PR 模板自动出现（7 条自查清单）
  - CI "check" 自动运行并变绿
  - 只有组长能点 Merge

---

## 2. Kaggle（数据 + 算力）

### 2.1 每人（含你）必做三件事

1. 注册 <https://www.kaggle.com>（**各人用自己的邮箱和手机**，不要一台电脑批量注册，容易被判滥用）；
2. `Settings → Phone Verification` 完成手机验证 —— **不验证拿不到 GPU**；
3. 把 Kaggle 用户名报给组长。

> 免费额度：约 **30 GPU 小时/周/账号**（T4×2 或 P100）。8 个账号 ≈ 240 小时/周，我们整个项目只要 8–12 小时。

### 2.2 你（组长）创建私有 Dataset

**方式一：网页拖拽（最简单，适合 <1 GB）**

1. 先把数据整理成本地这个结构：

```text
am-defect-v1/
├── raw/D1/defect/          # 136 张
├── raw/D1/nondefect/       # 135 张
├── raw/D2/defect/          # 947 张
├── raw/D2/nondefect/       # 2070 张
└── splits/                 # 地基组生成的 csv
```

2. Kaggle 首页 → `Datasets → New Dataset` → 拖文件夹进去 → 标题写 `am-defect-v1` → 可见性选 **Private** → Create

**方式二：命令行（推荐，可版本化）**

```powershell
pip install kaggle
# 先去 kaggle.com → Settings → API → Create New Token，下载 kaggle.json
# 放到 C:\Users\<你>\.kaggle\kaggle.json

kaggle datasets init -p am-defect-v1        # 生成 dataset-metadata.json
#   编辑它：title / id / licenses 三行
kaggle datasets create -p am-defect-v1 -r zip
kaggle datasets status <你的用户名>/am-defect-v1
# 以后更新：kaggle datasets version -p am-defect-v1 -m "v2: add splits"
```

> 上传限制：私有 Dataset 上限 200 GB，我们只有 ~1 GB，不用担心。首次上传约 10–30 分钟（看带宽）。

### 2.3 8 个人怎么用这份数据

在 Kaggle Notebook 里：

```python
import os
for root, dirs, files in os.walk("/kaggle/input/am-defect-v1"):
    print(root, len(files))
```

- 路径固定为 `/kaggle/input/am-defect-v1/`，**只读**；
- 所有输出写 `/kaggle/working/`；
- Notebook 默认**关网**：要 `git clone` 私有仓或装包时，右侧 `Settings → Internet → On`。

### 2.4 让 Notebook 能 clone 私有仓库（关键一步）

Kaggle 没法用你的浏览器登录态，需要一个 GitHub 访问令牌：

1. GitHub → `Settings → Developer settings → Personal access tokens → Fine-grained tokens` → 新建，勾 **只读** 权限（Contents: Read）→ 复制 `github_pat_xxx`；
2. Kaggle Notebook → `Add-ons → Secrets` → 新建 secret，名字写 `GH_TOKEN`，值粘贴 token；
3. Notebook 里：

```python
from kaggle_secrets import UserSecretsClient
import subprocess
tok = UserSecretsClient().get_secret("GH_TOKEN")
subprocess.run(f"git clone https://x-access-token:{tok}@github.com/Zheyi-D/am-defect-tl.git", shell=True)
```

---

## 3. Word Online（报告定稿）

1. 用 **HKU 邮箱**登录 <https://www.office.com> → Word → 新建空白文档；
2. 标题 `IDAT7215_Topic4_Report.docx`，存到 OneDrive 或组内共享文件夹；
3. 右上角 `共享` → 输入 7 个组员的 HKU 邮箱 → 权限选 **可编辑**；
4. 首页（第 1 页）直接写上：

```text
组号：______   班级：______
成员：姓名（学号）× 8
正文字数目标：3600（上限 4400，超 4400 扣 10%）

章节所有权
1 Introduction / Abstract / 4 Conclusion ...... 组长
2 Project Description ......................... 地基组
3 Results & Discussion ........................ 模型组 A/B/C + 分析

格式硬要求：A4 / 单倍行距 / Times New Roman 12 / IEEE 引用
```

5. 打开 `审阅 → 修订` 和版本历史，方便看谁改了什么。

> 为什么不用飞书/石墨做定稿：Moodle 只收 docx，飞书导出后字体/图表编号/字数统计都会乱。飞书只用来放大纲和讨论。

---

## 4. 飞书（沟通 + 看板）

沿用你们选题阶段已有的飞书环境，做三件事：

1. **建群** `IDAT7215 Topic4 小组`，8 人 + 组长；
2. **置顶一条消息**，包含 5 个链接：GitHub 仓 / Kaggle Dataset / Word Online / 飞书看板 / 本手册；
3. **多维表格** `Topic4 任务看板`，字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| 任务 | 文本 | 与 GitHub Issue 标题一致 |
| 组 | 单选 | 地基 / 模型A / 模型B / 模型C / 全员 |
| 负责人 | 人员 | |
| 截止 | 日期 | |
| 状态 | 单选 | 未开始 / 进行中 / 待 Review / 完成 |
| PR 链接 | 网址 | |
| run_id | 文本 | 对应 `results/runs/<run_id>` |

> GitHub Issue 管"技术任务"，飞书看板管"人的进度和卡点"，两边靠 `PR 链接` 字段对上。

---

## 5. 48 小时验收清单（逐条打勾）

- [ ] 7 个组员全部接受 GitHub 邀请，且 `git clone` 成功（截图发群）
- [ ] 至少 1 个组员成功提了一个 PR，CI 变绿，组长 merge 掉
- [ ] 8 人全部注册 Kaggle 并**完成手机验证**（截图能看到 GPU 选项）
- [ ] `am-defect-v1` 上传完成，8 人验证 `/kaggle/input/am-defect-v1/` 可读
- [ ] Word Online 文档建好，8 人能看到自己的章节标题
- [ ] 飞书群置顶消息 + 看板 8 条任务建好
- [ ] 组会开完，4 个组认领完毕
- [ ] 群里公布：第一次冒烟 deadline = **11/1**

---

## 6. 常见坑

| 现象 | 原因 | 解决 |
|---|---|---|
| `git push` 报 `Failed to connect to 127.0.0.1:7897` | 你的 git 全局配了代理，但 Clash/V2Ray 没开 | 开代理，或对本仓库执行 `git config --local --unset http.proxy` 和 `--unset https.proxy` |
| `gh` 能连但 `git` 不能 | 两者代理/凭据路径不同 | 用 GitHub Desktop，或按上一条处理代理 |
| GitHub 邀请点不开 | 7 天过期 | 组长重新 Add people |
| Kaggle 没有 GPU 选项 | 没做手机验证 | Settings → Phone Verification |
| Kaggle 上传卡住 | 网页上传大文件不稳 | 改用 `kaggle datasets create` 命令行 |
| Notebook `git clone` 失败 | 没开 Internet 或 token 没配对 | Settings → Internet → On；检查 Secret 名是否为 `GH_TOKEN` |
| 8 个人数据不一致 | 有人自己下载了数据 | 只允许挂载 `/kaggle/input/am-defect-v1`，禁止本地下载 |

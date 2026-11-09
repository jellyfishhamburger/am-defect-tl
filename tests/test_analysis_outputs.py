# Author: jellyfishhamburger | Task: T6 补测试 | PR: 待提
# Status: 本组原创实现。只用标准库，CI 不装 torch 也能跑。
"""T1/T2 产物的契约测试：数字必须与 results/ 一致，产物文件必须存在。

跑法：
    python -m unittest discover -s tests -v
    python -m unittest tests.test_analysis_outputs -v
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def leaderboard() -> list[dict]:
    p = RESULTS / "leaderboard.csv"
    if not p.exists():
        raise unittest.SkipTest("results/leaderboard.csv 不存在")
    with p.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def run_module(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout)


class TestExportedTables(unittest.TestCase):
    """T2：导出的表格必须逐格等于 leaderboard.csv。"""

    def test_tables_match_leaderboard(self):
        proc = run_module("-m", "analysis.export_tables", "--check")
        self.assertEqual(proc.returncode, 0,
                         msg=f"export_tables --check 失败\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("0 处不一致", proc.stdout)

    def test_each_table_has_markdown_and_latex(self):
        proc = run_module("-m", "analysis.export_tables")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])
        tables = sorted((ROOT / "reports" / "tables").glob("*.md"))
        self.assertGreaterEqual(len(tables), 5)
        for md in tables:
            tex = md.with_suffix(".tex")
            self.assertTrue(tex.exists(), f"{tex.name} 缺失")
            self.assertIn("\\begin{tabular}", tex.read_text(encoding="utf-8"))


class TestDashboard(unittest.TestCase):
    """看板：内嵌数据必须与 leaderboard.csv 一致，且页面零外链。"""

    def test_dashboard_embeds_all_runs(self):
        rows = leaderboard()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "index.html"
            proc = run_module("tools/build_dashboard.py", "--out", str(out))
            self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])
            html = out.read_text(encoding="utf-8")

        self.assertEqual(len(re.findall(r'src="https?://', html)), 0, "页面不应有外链脚本")
        self.assertEqual(len(re.findall(r'href="https?://', html)), 0, "页面不应有外链样式")

        m = re.search(r"const DATA = (\{.*?\});\n</script>", html, re.S)
        self.assertIsNotNone(m, "找不到内嵌的 DATA")
        data = json.loads(m.group(1))
        self.assertEqual(len(data["runs"]), len(rows))

        by_id = {r["run_id"]: r for r in rows}
        for run in data["runs"]:
            self.assertIn(run["id"], by_id)
            self.assertAlmostEqual(run["acc"]["test"], float(by_id[run["id"]]["accuracy"]), places=9)


class TestFigureIndex(unittest.TestCase):
    """T1：INDEX.md 里列的每张图都必须真的存在。"""

    def test_index_files_exist(self):
        index = RESULTS / "figures" / "INDEX.md"
        if not index.exists():
            self.skipTest("results/figures/INDEX.md 不存在（先跑 analysis.make_figures）")
        text = index.read_text(encoding="utf-8")
        linked = re.findall(r"\((fig_[A-Za-z0-9_\-]+\.png)\)", text)
        self.assertGreaterEqual(len(linked), 5, "INDEX.md 应至少列出 5 张图")
        for name in linked:
            self.assertTrue((RESULTS / "figures" / name).exists(), f"{name} 在 INDEX 里但文件不存在")


if __name__ == "__main__":
    unittest.main()

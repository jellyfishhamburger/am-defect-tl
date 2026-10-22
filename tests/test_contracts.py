# Author: 地基组 | Task: 契约测试（划分不泄漏 / metrics schema） | PR: #<编号>
# Status: 本组原创实现。仅用标准库，CI 里零依赖即可跑。
"""契约测试：保证 8 个人在同一份约定上跑。数据未就位时自动 skip。

跑法（二选一）：
    python -m unittest discover -s tests -v
    pytest tests/test_contracts.py -v
"""

from __future__ import annotations

import csv
import json
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "splits"
RUNS_DIR = ROOT / "results" / "runs"

SPLIT_COLS = ["path", "label", "split", "source_id"]
LABELS = {"defect", "nondefect"}
SPLIT_NAMES = {"train", "val", "test"}
EXPECTED_TOTAL = {"D1": 271, "D2": 3017}  # 见 docs/DATA.md §1
RATIO = {"train": 0.7, "val": 0.1, "test": 0.2}
TOL = 0.06

METRIC_NAMES = [
    "accuracy", "precision", "recall", "f1", "mcc", "kappa",
    "auc", "map", "sensitivity", "specificity",
]
METRICS_TOP_KEYS = ["run_id", "model", "dataset", "optimizer", "seed", "metrics", "confusion"]


def _read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


class TestSplits(unittest.TestCase):
    def _files(self) -> list[Path]:
        files = sorted(SPLITS_DIR.glob("*.csv"))
        if not files:
            self.skipTest("splits/*.csv 尚未生成（地基组任务，10/29 前）")
        return files

    def test_schema(self):
        for path in self._files():
            header, rows = _read_rows(path)
            with self.subTest(file=path.name):
                self.assertEqual(header, SPLIT_COLS, f"{path.name} 表头必须严格为 {SPLIT_COLS}")
                self.assertTrue(rows, f"{path.name} 不能为空")
                for r in rows:
                    self.assertIn(r["label"], LABELS, f"非法 label: {r['label']}")
                    self.assertIn(r["split"], SPLIT_NAMES, f"非法 split: {r['split']}")

    def test_all_three_splits_present(self):
        for path in self._files():
            _, rows = _read_rows(path)
            with self.subTest(file=path.name):
                self.assertEqual({r["split"] for r in rows}, SPLIT_NAMES)

    def test_ratio(self):
        for path in self._files():
            _, rows = _read_rows(path)
            with self.subTest(file=path.name):
                counts = Counter(r["split"] for r in rows)
                total = len(rows)
                for split, expected in RATIO.items():
                    actual = counts[split] / total
                    self.assertLess(abs(actual - expected), TOL,
                                    f"{path.name} 的 {split} 占比 {actual:.2f} 偏离 {expected}")

    def test_paths_disjoint(self):
        for path in self._files():
            _, rows = _read_rows(path)
            with self.subTest(file=path.name):
                by_split = {s: {r["path"] for r in rows if r["split"] == s} for s in SPLIT_NAMES}
                self.assertFalse(by_split["train"] & by_split["test"], "train 与 test 有重叠样本")
                self.assertFalse(by_split["train"] & by_split["val"], "train 与 val 有重叠样本")
                self.assertFalse(by_split["val"] & by_split["test"], "val 与 test 有重叠样本")

    def test_group_no_leak(self):
        """核心：同一 source_id 的增强副本不得跨 split。"""
        for path in self._files():
            if "group" not in path.stem:
                continue  # 只有分组划分才要求 source_id
            _, rows = _read_rows(path)
            with self.subTest(file=path.name):
                where = {}
                for r in rows:
                    sid = r["source_id"]
                    self.assertTrue(sid, f"{path.name} 里 source_id 不能为空")
                    where.setdefault(sid, set()).add(r["split"])
                leaked = {sid: s for sid, s in where.items() if len(s) > 1}
                self.assertFalse(leaked, f"以下 source_id 跨 split（数据泄漏）: {list(leaked)[:5]}")

    def test_expected_total(self):
        for path in self._files():
            dim = path.stem
            if dim not in EXPECTED_TOTAL:
                continue
            _, rows = _read_rows(path)
            with self.subTest(file=path.name):
                self.assertEqual(len(rows), EXPECTED_TOTAL[dim],
                                 f"{path.name} 应有 {EXPECTED_TOTAL[dim]} 行（见 docs/DATA.md §1）")


class TestRunArtifacts(unittest.TestCase):
    def _run_files(self) -> list[Path]:
        files = sorted(RUNS_DIR.glob("*/metrics.json"))
        if not files:
            self.skipTest("results/runs/*/metrics.json 尚未产出")
        return files

    def test_metrics_schema(self):
        for path in self._run_files():
            data = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(run=path.parent.name):
                for key in METRICS_TOP_KEYS:
                    self.assertIn(key, data, f"缺少顶层字段 {key}")
                self.assertEqual(data["run_id"], path.parent.name, "run_id 必须与目录名一致")
                for split in SPLIT_NAMES:
                    self.assertIn(split, data["metrics"], f"缺少 {split} 指标")
                    for name in METRIC_NAMES:
                        value = data["metrics"][split].get(name)
                        self.assertIsNotNone(value, f"{split}.{name} 缺失")
                        self.assertGreaterEqual(value, 0.0, f"{split}.{name} 越界")
                        self.assertLessEqual(value, 1.0, f"{split}.{name} 越界")

    def test_confusion_shape(self):
        for path in self._run_files():
            data = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(run=path.parent.name):
                for split in SPLIT_NAMES:
                    cm = data["confusion"][split]
                    self.assertEqual(len(cm), 2, f"{split} 混淆矩阵应为 2 行")
                    self.assertTrue(all(len(row) == 2 for row in cm), f"{split} 每行应 2 列")
                    self.assertEqual(sum(sum(row) for row in cm), data.get(f"n_{split}", sum(sum(row) for row in cm)))


class TestCoverage(unittest.TestCase):
    def test_run_coverage_progress(self):
        """只报告进度，不阻断 PR（66 个 run 分三周跑完）。"""
        n = len(list(RUNS_DIR.glob("*/metrics.json")))
        if n == 0:
            self.skipTest("尚无 run")
        print(f"\n[progress] 已完成 run: {n}/66")
        self.assertLessEqual(n, 66 * 2, "run 数量异常偏多，检查是否有重复 run_id")


if __name__ == "__main__":
    unittest.main(verbosity=2)

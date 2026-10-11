"""Check that inherited debt cannot excuse new files, rules, or higher counts."""

import unittest
from pathlib import Path

from check_ruff_baseline import regressions


class BaselineTests(unittest.TestCase):
    def test_existing_and_reduced_debt_pass(self):
        root = Path("/checkout")
        row = {"filename": "/checkout/src/example.py", "code": "F401"}
        baseline = {"src/example.py": {"F401": 2}}
        self.assertEqual(regressions([row, row], baseline, root), {})
        self.assertEqual(regressions([row], baseline, root), {})

    def test_new_file_rule_and_count_fail(self):
        root = Path("/checkout")
        row = {"filename": "/checkout/src/example.py", "code": "F401"}
        baseline = {"src/example.py": {"F401": 1}}
        for rows, expected in (
            ([row, row], {("src/example.py", "F401"): 1}),
            ([dict(row, code="F821")], {("src/example.py", "F821"): 1}),
            ([dict(row, filename="/checkout/new.py")], {("new.py", "F401"): 1}),
        ):
            with self.subTest(rows=rows):
                self.assertEqual(regressions(rows, baseline, root), expected)


if __name__ == "__main__":
    unittest.main()

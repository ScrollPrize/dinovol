from __future__ import annotations

import unittest

from dinovol_2.ops.weighted_loader import WeightedCombinedLoader


class _CountingLoader:
    def __init__(self, value: str) -> None:
        self.value = value
        self.iter_calls = 0
        self.sampler = None
        self.dataset = None

    def __iter__(self):
        self.iter_calls += 1
        return iter((self.value,))


class WeightedCombinedLoaderTests(unittest.TestCase):
    def test_source_iterators_are_created_only_when_selected(self) -> None:
        first = _CountingLoader("first")
        second = _CountingLoader("second")
        combined = WeightedCombinedLoader((first, second), weights=(0.95, 0.05), seed=0)

        iterator = iter(combined)
        self.assertEqual(first.iter_calls, 0)
        self.assertEqual(second.iter_calls, 0)

        self.assertEqual(next(iterator), "first")
        self.assertEqual(first.iter_calls, 1)
        self.assertEqual(second.iter_calls, 0)


if __name__ == "__main__":
    unittest.main()

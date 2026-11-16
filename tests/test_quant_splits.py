from __future__ import annotations

from datetime import date, timedelta
import unittest

from quant_research.splits import make_splits


class TemporalSplitTests(unittest.TestCase):
    def setUp(self):
        self.dates = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(12))

    def test_excludes_label_horizon_at_both_boundaries(self):
        split = make_splits(
            self.dates, train_end=self.dates[2], dev_start=self.dates[5],
            dev_end=self.dates[7], holdout_start=self.dates[10], label_horizon=2,
        )
        self.assertEqual(split.train, self.dates[:3])
        self.assertEqual(split.dev, self.dates[5:8])
        self.assertEqual(split.holdout, self.dates[10:])
        self.assertEqual(split.excluded, self.dates[3:5] + self.dates[8:10])

    def test_rejects_short_gap_and_unordered_boundaries(self):
        with self.assertRaisesRegex(ValueError, "train/dev gap"):
            make_splits(self.dates, train_end=self.dates[2], dev_start=self.dates[4],
                        dev_end=self.dates[7], holdout_start=self.dates[10], label_horizon=2)
        with self.assertRaisesRegex(ValueError, "chronological"):
            make_splits(self.dates, train_end=self.dates[5], dev_start=self.dates[4],
                        dev_end=self.dates[7], holdout_start=self.dates[10])

    def test_rejects_bad_calendar_or_horizon(self):
        kwargs = dict(train_end=self.dates[2], dev_start=self.dates[5],
                      dev_end=self.dates[7], holdout_start=self.dates[10])
        for dates in (self.dates[::-1], self.dates + (self.dates[-1],)):
            with self.assertRaisesRegex(ValueError, "unique and increasing"):
                make_splits(dates, **kwargs)
        for horizon in (0, True, 1.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                make_splits(self.dates, label_horizon=horizon, **kwargs)


if __name__ == "__main__":
    unittest.main()

"""Chronological, non-overlapping windows for factor research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class TemporalSplit:
    train: tuple[date, ...]
    dev: tuple[date, ...]
    holdout: tuple[date, ...]
    excluded: tuple[date, ...]
    label_horizon: int


def make_splits(
    dates: tuple[date, ...], *, train_end: date, dev_start: date,
    dev_end: date, holdout_start: date, label_horizon: int = 1,
) -> TemporalSplit:
    """Split a calendar, excluding at least one label horizon at each boundary.

    A label formed at index ``i`` may depend on prices through ``i + H``.
    The first date of a later window must be farther than H sessions from
    the last date of the previous window, so no labeled observation crosses.
    """
    if not dates or any(type(day) is not date for day in dates):
        raise ValueError("dates must be nonempty date objects")
    if tuple(sorted(set(dates))) != dates:
        raise ValueError("dates must be unique and increasing")
    if type(label_horizon) is not int or label_horizon < 1:
        raise ValueError("label_horizon must be a positive integer")
    try:
        train_last = dates.index(train_end)
        dev_first = dates.index(dev_start)
        dev_last = dates.index(dev_end)
        holdout_first = dates.index(holdout_start)
    except ValueError as exc:
        raise ValueError("split boundaries must be trading dates") from exc
    if not (train_last < dev_first <= dev_last < holdout_first):
        raise ValueError("split boundaries must be chronological and nonempty")
    if dev_first - train_last - 1 < label_horizon:
        raise ValueError("train/dev gap is shorter than label_horizon")
    if holdout_first - dev_last - 1 < label_horizon:
        raise ValueError("dev/holdout gap is shorter than label_horizon")
    return TemporalSplit(
        train=dates[:train_last + 1],
        dev=dates[dev_first:dev_last + 1],
        holdout=dates[holdout_first:],
        excluded=dates[train_last + 1:dev_first] + dates[dev_last + 1:holdout_first],
        label_horizon=label_horizon,
    )

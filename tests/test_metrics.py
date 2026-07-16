"""Unit tests for continual-learning metrics."""

from __future__ import annotations

import pytest

from train_utils import backward_transfer, forgetting


def test_forgetting_two_task():
    """Forgetting is positive when old-task error increases; averaged over all tasks."""
    nmaes = [[1.0], [1.5, 2.0]]
    # Task 0: 1.5 - 1.0 = 0.5; task 1: 2.0 - 2.0 = 0.0; average = 0.25.
    assert forgetting(nmaes) == pytest.approx(0.25)


def test_forgetting_no_forgetting():
    """Forgetting is zero when old-task error never increases."""
    nmaes = [[1.0], [1.0, 2.0]]
    assert forgetting(nmaes) == pytest.approx(0.0)


def test_forgetting_multi_task():
    """Forgetting averages per-task forgetting over all tasks."""
    nmaes = [[1.0], [1.2, 2.0], [1.5, 1.8, 3.0]]
    # Task 0: best=1.0, final=1.5 -> 0.5
    # Task 1: best=2.0, final=1.8 -> 0.0 (clipped)
    # Task 2: best=3.0, final=3.0 -> 0.0
    assert forgetting(nmaes) == pytest.approx(0.5 / 3)


def test_bwt_two_task_forgetting():
    """BWT is negative for error-metric forgetting."""
    nmaes = [[1.0], [1.5, 2.0]]
    assert backward_transfer(nmaes) == pytest.approx(-0.5)


def test_bwt_two_task_positive_transfer():
    """BWT is positive when old-task error decreases after learning a new task."""
    nmaes = [[1.0], [0.8, 2.0]]
    assert backward_transfer(nmaes) == pytest.approx(0.2)


def test_bwt_equals_negative_forgetting_two_tasks():
    """For two tasks, BWT equals -absolute_forgetting."""
    nmaes = [[1.0], [1.7, 2.0]]
    abs_forgetting = nmaes[1][0] - nmaes[0][0]
    assert backward_transfer(nmaes) == pytest.approx(-abs_forgetting)


def test_bwt_multi_task_denominator():
    """BWT averages over T-1 old tasks."""
    nmaes = [[1.0], [1.2, 2.0], [1.5, 1.8, 3.0]]
    # Task 0: best=1.0, final=1.5 -> -0.5
    # Task 1: best=2.0, final=1.8 -> +0.2
    expected = (-0.5 + 0.2) / 2
    assert backward_transfer(nmaes) == pytest.approx(expected)


def test_bwt_single_task_is_zero():
    """BWT and forgetting are zero with only one task."""
    nmaes = [[1.0]]
    assert backward_transfer(nmaes) == pytest.approx(0.0)
    assert forgetting(nmaes) == pytest.approx(0.0)

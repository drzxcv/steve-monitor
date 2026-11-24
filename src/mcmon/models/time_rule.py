"""Baseline that only looks at elapsed time: predict 'complete' after a per-task step."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..metrics import prf

if TYPE_CHECKING:
    from ..data import Samples


class TimeRule:
    """One step cutoff per task, chosen for the best F1."""

    def fit(self, samples: Samples, n_tasks: int) -> TimeRule:
        self.cutoff = np.zeros(n_tasks)
        for tid in range(n_tasks):
            m = samples.task == tid
            candidates = np.unique(samples.step[m])
            scores = [prf(samples.y[m], samples.step[m] >= c)["f1"] for c in candidates]
            self.cutoff[tid] = candidates[int(np.argmax(scores))] if len(candidates) else 0
        return self

    def predict_proba(self, samples: Samples) -> np.ndarray:
        """1.0 at or after the cutoff, else 0.0."""
        return (samples.step >= self.cutoff[samples.task]).astype(np.float32)

"""Gradient-boosted trees on the flattened window."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xgboost as xgb

if TYPE_CHECKING:
    from ..data import Samples


class TreeMonitor:
    """XGBoost classifier with early stopping on validation."""

    def __init__(self, seed: int = 0, **params: Any):
        defaults = dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                        colsample_bytree=0.5, tree_method="hist", early_stopping_rounds=20)
        self.model = xgb.XGBClassifier(**{**defaults, **params}, random_state=seed)

    def fit(self, train: Samples, val: Samples) -> TreeMonitor:
        self.model.fit(train.flat(), train.y, eval_set=[(val.flat(), val.y)], verbose=False)
        return self

    def predict_proba(self, samples: Samples) -> np.ndarray:
        """Probability of completion for each sample."""
        return self.model.predict_proba(samples.flat())[:, 1]

from __future__ import annotations

import numpy as np
from aeon.classification.convolution_based import MiniRocketClassifier


class MiniRocket:
    def __init__(
        self,
        random_state: int | None = 42,
        num_kernels: int = 10000,
        **kwargs,
    ):
        self.random_state = random_state
        self.num_kernels = num_kernels
        self.kwargs = kwargs
        self.model = self._build_model()

    def _build_model(self):
        return MiniRocketClassifier(
            n_kernels=self.num_kernels,
            n_jobs=-1,
            random_state=self.random_state,
            **self.kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
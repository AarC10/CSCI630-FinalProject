from aeon.classification.convolution_based import MiniRocketClassifier
from sklearn.linear_model import LogisticRegression as SklearnLR
import numpy as np


class MiniRocket:
    def __init__(
        self,
        random_state: int | None = 42,
        num_kernels: int = 1000,
        C: float = 1.0,
        max_iter: int = 3000,
        class_weight: str | None = None,
        **kwargs,
    ):
        self.random_state = random_state
        self.num_kernels = num_kernels
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.kwargs = kwargs
        self.model = self._build_model()

    def _build_model(self):
        estimator = SklearnLR(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        return MiniRocketClassifier(
            n_kernels=self.num_kernels,
            estimator=estimator,
            n_jobs=-1,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
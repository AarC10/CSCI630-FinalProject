import numpy as np
from aeon.classification.feature_based import Catch22Classifier
from sklearn.linear_model import LogisticRegression


class Catch22LogisticRegression:
    def __init__(
        self,
        random_state: int | None = 42,
        max_iter: int = 1000,
        C: float = 1.0,
        solver: str = "saga",
        **kwargs,
    ):
        self.random_state = random_state
        self.max_iter = max_iter
        self.C = C
        self.solver = solver
        self.kwargs = kwargs
        self.model = self._build_model()

    def _build_model(self):
        return Catch22Classifier(
            estimator=LogisticRegression(
                random_state=self.random_state,
                max_iter=self.max_iter,
                C=self.C,
                solver=self.solver,
                **self.kwargs,
            ),
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
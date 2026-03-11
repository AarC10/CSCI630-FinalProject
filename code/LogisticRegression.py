import numpy as np

from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegression:
    def __init__(
        self,
        random_state: int | None = 42,
        max_iter: int = 1000,
        C: float = 1.0,
        solver: str = "lbfgs",
        **kwargs,
    ):
        self.random_state = random_state
        self.max_iter = max_iter
        self.C = C
        self.solver = solver
        self.kwargs = kwargs
        self.model = self._build_model()

    def _build_model(self):
        return make_pipeline(
            StandardScaler(),
            SklearnLogisticRegression(
                random_state=self.random_state,
                max_iter=self.max_iter,
                C=self.C,
                solver=self.solver,
                class_weight="balanced",
                **self.kwargs,
            ),
        )

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (n_samples, n_channels, n_timesteps). Got {X.shape}.")
        return X.reshape(X.shape[0], -1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(self._flatten(X), np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._flatten(X))

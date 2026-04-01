import numpy as np

from aeon.classification.interval_based import TimeSeriesForestClassifier as AeonTSF


class TimeSeriesForest:
    def __init__(
            self,
            random_state: int | None = 42,
            n_estimators: int = 200,
            min_interval_length: int = 3,
            n_jobs: int = -1,
            **kwargs,
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.min_interval_length = min_interval_length
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.model = self._build_model()

    def _build_model(self):
        return AeonTSF(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            min_interval_length=self.min_interval_length,
            n_jobs=self.n_jobs,
            **self.kwargs,
        )

    def _validate(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (n_samples, n_channels, n_timesteps). Got {X.shape}.")
        return X

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(self._validate(X), np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._validate(X))
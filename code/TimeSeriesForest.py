import numpy as np

from sktime.classification.interval_based import TimeSeriesForestClassifier as SklearnTSF
from sktime.datatypes import convert


class TimeSeriesForest:
    def __init__(
            self,
            random_state: int | None = 42,
            n_estimators: int = 200,
            min_interval: int = 3,
            n_jobs: int = -1,
            **kwargs,
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.min_interval = min_interval
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.model = self._build_model()

    def _build_model(self):
        return SklearnTSF(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            min_interval=self.min_interval,
            n_jobs=self.n_jobs,
            **self.kwargs,
        )

    def _convert(self, X: np.ndarray):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (n_samples, n_channels, n_timesteps). Got {X.shape}.")
        # sktime expects (n_samples, n_channels, n_timesteps) as a nested dataframe
        return convert(X, from_type="numpy3D", to_type="nested_univ")

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(self._convert(X), np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._convert(X))
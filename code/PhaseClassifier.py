from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Optional

import numpy as np
import logging
from sklearn.metrics import accuracy_score, f1_score

from LogisticRegression import LogisticRegression
from Catch22 import Catch22LogisticRegression
from MiniRocket import MiniRocket
from Knn import KNN
from TimeSeriesForest import TimeSeriesForest

logger = logging.getLogger(__name__)

class FlightPhaseClassifier:
    SUPPORTED_MODEL_TYPES = {"lr", "knn", "tsf", "minirocket", "catch22_lr"}
    PHASE_MAPPING = {
        0: "no_event",
        1: "liftoff",
        2: "burnout",
        3: "apogee",
        4: "recovery_deployment",
        # Excluding landing since most of the dataset hasn't labelled this for some reason :P
        # 5: "landing",
    }

    def __init__(self, model_type: str, random_state: Optional[int] = 42, **model_kwargs: Any):
        if model_type not in self.SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Expected one of: {sorted(self.SUPPORTED_MODEL_TYPES)}"
            )

        self.model_type = model_type
        self.random_state = random_state
        self.model_kwargs = dict(model_kwargs)
        self.model: Any | None = None
        self.is_fitted = False
        self.train_time = 0.0
        self.last_inference_time = 0.0

    def _build_model(self) -> Any:
        if self.model_type == "lr":
            return LogisticRegression(random_state=self.random_state, **self.model_kwargs)

        if self.model_type == "catch22_lr":
            return Catch22LogisticRegression(random_state=self.random_state, **self.model_kwargs)

        if self.model_type == "knn":
            return KNN()

        if self.model_type == "tsf":
            return TimeSeriesForest(random_state=self.random_state, **self.model_kwargs)

        if self.model_type == "minirocket":
            return MiniRocket(random_state=self.random_state, **self.model_kwargs)

        raise RuntimeError(f"Unhandled model_type: {self.model_type}")

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        if X.ndim != 3:
            raise ValueError(
                "X must have shape (n_samples, n_channels, n_timesteps). "
                f"Got shape {X.shape}."
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if X.shape[1] == 0 or X.shape[2] == 0:
            raise ValueError("X must have at least one channel and one timestep.")
        if np.isnan(X).any():
            raise ValueError("X contains NaN values. Clean or filter windows before fitting.")

        return X

    def _validate_y(self, y: np.ndarray, n_samples: Optional[int] = None) -> np.ndarray:
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError(f"y must be a 1D array of labels. Got shape {y.shape}.")
        if y.size == 0:
            raise ValueError("y must contain at least one label.")
        if n_samples is not None and len(y) != n_samples:
            raise ValueError(f"X and y must contain the same number of samples. Got {n_samples} and {len(y)}.")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must contain integer flight phase labels.")

        unknown_labels = sorted(set(np.unique(y).tolist()) - set(self.PHASE_MAPPING.keys()))
        if unknown_labels:
            raise ValueError(
                f"y contains unsupported flight phase labels: {unknown_labels}. "
                f"Expected labels from {sorted(self.PHASE_MAPPING.keys())}."
            )

        return y.astype(np.int64, copy=False)

    def _ensure_model(self) -> Any:
        if self.model is None:
            self.model = self._build_model()
        return self.model

    def _ensure_fitted(self) -> None:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Classifier has not been fitted yet. Call fit(X, y) first.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PhaseClassifier":
        X = self._validate_X(X)
        y = self._validate_y(y, n_samples=X.shape[0])

        model = self._ensure_model()

        start = perf_counter()
        model.fit(X, y)
        self.train_time = perf_counter() - start
        self.is_fitted = True

        logger.info(f"Finished fitting {self.model_type} model in {self.train_time:.2f} seconds.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        X = self._validate_X(X)

        start = perf_counter()
        predictions = self.model.predict(X)
        self.last_inference_time = perf_counter() - start

        logger.info(f"Finished predicting {len(X)} samples in {self.last_inference_time:.2f} seconds.")
        return np.asarray(predictions, dtype=np.int64)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        X = self._validate_X(X)
        y = self._validate_y(y, n_samples=X.shape[0])

        predictions = self.predict(X)
        labels = list(self.PHASE_MAPPING.keys())
        per_class_scores = f1_score(y, predictions, labels=labels, average=None, zero_division=0)

        return {
            "accuracy": float(accuracy_score(y, predictions)),
            "weighted_f1": float(f1_score(y, predictions, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
            "per_class_f1": {label: float(score) for label, score in zip(labels, per_class_scores)},
            "train_time": float(self.train_time),
            "inference_time": float(self.last_inference_time),
        }

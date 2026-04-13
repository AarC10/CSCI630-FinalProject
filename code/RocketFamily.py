import inspect

import numpy as np

from aeon.classification.convolution_based import (
    HydraClassifier,
    MultiRocketClassifier,
    MultiRocketHydraClassifier,
)


class _AeonConvClassifierWrapper:
    classifier_cls = None

    def __init__(
        self,
        random_state: int | None = 42,
        n_jobs: int = -1,
        class_weight: str | None = None,
        num_kernels: int | None = None,
        n_kernels: int | None = None,
        **kwargs,
    ):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.n_kernels = num_kernels if num_kernels is not None else n_kernels
        self.kwargs = kwargs
        self.model = self._build_model()

    def _build_model(self):
        if self.classifier_cls is None:
            raise ValueError("classifier_cls must be set on subclasses.")

        signature = inspect.signature(self.classifier_cls.__init__)
        candidate_kwargs = {
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "class_weight": self.class_weight,
            "num_kernels": self.n_kernels,
            "n_kernels": self.n_kernels,
            **self.kwargs,
        }

        filtered_kwargs = {
            key: value for key, value in candidate_kwargs.items()
            if value is not None and key in signature.parameters
        }

        if "num_kernels" in filtered_kwargs and "n_kernels" in filtered_kwargs:
            filtered_kwargs.pop("num_kernels")

        return self.classifier_cls(**filtered_kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class MultiRocket(_AeonConvClassifierWrapper):
    classifier_cls = MultiRocketClassifier


class Hydra(_AeonConvClassifierWrapper):
    classifier_cls = HydraClassifier


class MultiRocketHydra(_AeonConvClassifierWrapper):
    classifier_cls = MultiRocketHydraClassifier

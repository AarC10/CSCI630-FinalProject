from aeon.classification.convolution_based import RocketClassifier
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.linear_model import RidgeClassifierCV
import numpy as np


class MiniRocket:
    def __init__(
        self,
        random_state: int | None = 42,
        num_kernels: int = 1000,
        max_dilations_per_kernel: int = 32,
        C: float = 1.0,
        max_iter: int = 3000,
        class_weight: str | None = "balanced",
        estimator_name: str = "ridge",
        ridge_alphas: list[float] | tuple[float, ...] | np.ndarray | None = None,
        n_jobs: int = -1,
        **kwargs,
    ):
        self.random_state = random_state
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.estimator_name = estimator_name.lower()
        self.ridge_alphas = ridge_alphas
        self.n_jobs = n_jobs
        self.solver = kwargs.pop("solver", "lbfgs")
        self.tol = kwargs.pop("tol", 1e-4)
        self.kwargs = kwargs
        self.model = self._build_model()

    def _build_estimator(self):
        if self.estimator_name == "ridge":
            if self.ridge_alphas is None:
                return None
            return RidgeClassifierCV(
                alphas=np.asarray(self.ridge_alphas, dtype=float),
                class_weight=self.class_weight,
            )

        if self.estimator_name == "logistic":
            return SklearnLR(
                C=self.C,
                max_iter=self.max_iter,
                solver=self.solver,
                class_weight=self.class_weight,
                random_state=self.random_state,
                tol=self.tol,
                multi_class="auto",
                **self.kwargs,
            )

        raise ValueError(
            f"Unsupported MiniRocket estimator_name '{self.estimator_name}'. "
            "Expected one of: ['ridge', 'logistic']."
        )

    def _build_model(self):
        estimator = self._build_estimator()
        wrapper_class_weight = self.class_weight if self.estimator_name == "ridge" and estimator is None else None
        return RocketClassifier(
            num_kernels=self.num_kernels,
            rocket_transform="minirocket",
            max_dilations_per_kernel=self.max_dilations_per_kernel,
            estimator=estimator,
            class_weight=wrapper_class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

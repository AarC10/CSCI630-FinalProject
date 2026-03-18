#!/usr/bin/env python3
"""
Example script for training event classifiers on rocket flight data.
This demonstrates how to use the generated datasets for event classification.
"""
import pandas as pd
import numpy as np


from sklearn.neighbors import KNeighborsRegressor # Use KNeighborsRegressor for regression KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


from sklearn.pipeline import make_pipeline
from scipy.spatial import distance

class KNN:
    #n_neighbors = [1,2,3,4,5]

    def __init__(
           self,
        random_state: int | None = 42,
        max_iter: int = 1000,
        C: float = 1.0,
        solver: str = "lfbgs",
        **kwargs,
    ):
        self.random_state = random_state
        self.max_iter = max_iter
        self.C = C
        self.solver = solver
        self.kwargs = kwargs
        self.model = self._build_model()

    
    def dtw_metric(a , b):
         # Implementation of the DTW distance calculation
        an = a.size
        bn = b.size
        pointwise_distance = distance.cdist(a.reshape(-1, 1), b.reshape(-1, 1))
        cumdist = np.matrix(np.ones((an + 1, bn + 1)) * np.inf)
        cumdist[0, 0] = 0
        for ai in range(an):
            for bi in range(bn):
                minimum_cost = np.min([cumdist[ai, bi+1], cumdist[ai+1, bi], cumdist[ai, bi]])
                cumdist[ai+1, bi+1] = pointwise_distance[ai, bi] + minimum_cost
        return cumdist[an, bn]

    def _build_model(self):
        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors = self.n_neighbors, metric = self.dtw_metric)
        )
    
    def fit (self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X,y )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
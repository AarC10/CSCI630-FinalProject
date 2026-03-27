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
        n_neighbors: int | None = 5,
        weights: str = "uniform",
        metric: str = "minkowski",
       # **kwargs,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
       # self.kwargs = kwargs
        self.model=self._build_model()

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
            KNeighborsRegressor(n_neighbors=self.n_neighbors,
                                 weights=self.weights,
                                 metric=self.metric,
                                 n_jobs=1,
                                 )
        )
    
    def _flatten(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return X.reshape(X.shape[0], -1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(self._flatten(X), np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._flatten(X))
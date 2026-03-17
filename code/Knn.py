#!/usr/bin/env python3
"""
Example script for training event classifiers on rocket flight data.
This demonstrates how to use the generated datasets for event classification.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor # Use KNeighborsRegressor for regression KNeighborsClassifier
from sklearn.impute import KNNImputer 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from scipy.spatial import distance

class KNN:
    #n_neighbors = [1,2,3,4,5]

    def __init__(
            self,
            n_neighbor: int | None = 5,
    ):
        self.n_neighbors = [1,2,3,4,5]
        #self.metric = dtw_metric
        self.mode
    
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

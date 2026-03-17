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



class Test:

    def __init__(self, data_dir: str = "./dataset"):
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
        self.models = {}
        self.event_types = ['liftoff', 'burnout', 'apogee', 'recovery_deployment', 'landing']


    def load_data(self, dataset_type: str = "features_generated"):
        """Load the generated rocket flight data with events and features."""
        data_path = self.data_dir / dataset_type

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset directory {data_path} not found. "
                                    "Run the data generator first with include_event_labels=True "
                                    "and include_features=True")

        # Load all CSV files
        all_data = []
        csv_files = list(data_path.glob("*.csv"))
        print(f"Loading {len(csv_files)} files from {data_path}")

        for csv_file in csv_files[:10]:  # Limit to first 10 files for demo
            df = pd.read_csv(csv_file)
            all_data.append(df)

        if not all_data:
            raise ValueError(f"No CSV files found in {data_path}")

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined_data)} total samples")

        imputer = KNNImputer(n_neighbors=5)
        combined_data_fill = imputer.fit_transform(combined_data)
        combined_data_fill = pd.DataFrame(combined_data_fill, columns=df.columns) 

        return combined_data_fill
    
    def prepare_features_and_labels(self, df: pd.DataFrame):
        """Separate features from event labels."""
        # Identify event columns (no more upcoming variants to filter out)
        event_cols = [col for col in df.columns if col.startswith('event_')]

        # Feature columns (exclude event columns only)
        feature_cols = [col for col in df.columns
                       if not col.startswith('event_')]

        X = df[feature_cols].copy()
        y = df[event_cols].copy()

        print(f"Features: {len(feature_cols)} columns")
        print(f"Event types: {event_cols}")
        print(f"Event distribution:")
        for col in event_cols:
            event_count = y[col].sum()
            total_count = len(y)
            print(f"  {col}: {event_count}/{total_count} ({100*event_count/total_count:.2f}%)")

        return X, y, feature_cols, event_cols
    
    def train_single_event_classifier(self, X: pd.DataFrame, y: pd.Series, event_name: str):
        """Train a binary classifier for a single event type."""
        # Handle class imbalance
        if y.sum() == 0:
            print(f"Warning: No positive examples for {event_name}, skipping")
            return None

        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = dict(zip(np.unique(y), class_weights))

        # Train Random Forest
        """ rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=class_weight_dict,
            random_state=42
        ) """

        #Train KNN
        knn = KNeighborsRegressor(n_neighbors=5)
        

        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Train and evaluate
            #rf.fit(X_train_scaled, y_train)
            knn.fit(X, y)
            #y_pred = rf.predict(X_val_scaled)
            y_pred = knn.predict(X)

            if len(np.unique(y_val)) > 1:  # Only calculate F1 if both classes present
                f1 = f1_score(y_val, y_pred)
                scores.append(f1)

        if scores:
            avg_f1 = np.mean(scores)
            print(f"{event_name} - Average F1 Score: {avg_f1:.3f}")

        # Train final model on all data
        X_scaled = self.scaler.fit_transform(X)
        #rf.fit(X_scaled, y)
        knn.fit(X_scaled,y)

        #return rf
        return knn
    
    def train_all_classifiers(self, dataset_type: str = "features_generated"):
        """Train classifiers for all event types."""
        # Load data
        df = self.load_data(dataset_type)
        X, y, feature_cols, event_cols = self.prepare_features_and_labels(df)

        # Train individual classifiers for each event type
        print("\nTraining classifiers...")
        for event_col in event_cols:
            event_name = event_col.replace('event_', '')
            print(f"\nTraining classifier for {event_name}")

            model = self.train_single_event_classifier(X, y[event_col], event_name)
            if model:
                self.models[event_name] = {
                    'model': model,
                    'scaler': self.scaler,
                    'feature_cols': feature_cols
                }
    
    def predict_events(self, X: pd.DataFrame):
        """Predict events for new data."""
        predictions = {}

        for event_name, model_data in self.models.items():
            model = model_data['model']
            scaler = model_data['scaler']
            feature_cols = model_data['feature_cols']

            # Select and scale features
            X_features = X[feature_cols]
            X_scaled = scaler.transform(X_features)

            # Predict
            y_pred = model.predict(X_scaled)
            y_proba = model.predict_proba(X_scaled)[:, 1]  # Probability of positive class

            predictions[event_name] = {
                'predictions': y_pred,
                'probabilities': y_proba
            }
        return predictions

    def evaluate_model(self, dataset_type: str = "features_ground_truth"):
        """Evaluate trained models on test data."""
        print(f"\nEvaluating on {dataset_type} dataset...")

        # Load test data
        df_test = self.load_data(dataset_type)
        X_test, y_test, _, event_cols = self.prepare_features_and_labels(df_test)

        # Make predictions
        predictions = self.predict_events(X_test)
        print("predictions", predictions, '\n')

        # Evaluate each event type
        for event_col in event_cols:
            event_name = event_col.replace('event_', '')
            print("\n event_name:", event_name, '\n')

            if event_name in predictions:
                y_true = y_test[event_col]
                y_pred = predictions[event_name]['predictions']

                print(f"\n{event_name.upper()} Classification Report:")
                print(classification_report(y_true, y_pred))

    def plot_feature_importance(self, event_name: str, top_n: int = 15):
        """Plot feature importance for a specific event classifier."""
        if event_name not in self.models:
            print(f"No model found for {event_name}")
            return

        model = self.models[event_name]['model']
        feature_cols = self.models[event_name]['feature_cols']

        # Get feature importances
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Plot top N features
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance_df.head(top_n), x='importance', y='feature')
        plt.title(f'Top {top_n} Features for {event_name.title()} Detection')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.show()

    



RocketType = ["Barry", "Flds", "Kong", "L1", "L2", "L9",
                "LilBro", "MayFlower","May", "Eagle",
                "Olym", "Omen", "Pro", "Risk", "Tua", "Void"]


Object = Test()
for name in RocketType:
   #load and train data 
   Object.train_all_classifiers(name)

   # Evaluate on clean data to see how well it generalizes
   Object.evaluate_model(name)

# Plot feature importance for apogee detection
   Object.plot_feature_importance("apogee")

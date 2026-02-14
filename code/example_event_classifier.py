#!/usr/bin/env python3
"""
Example script for training event classifiers on rocket flight data.
This demonstrates how to use the generated datasets for event classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

class EventClassifier:
    def __init__(self, data_dir: str = "./simulation_data"):
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
        print(f"Loading {len(csv_files)} files from {dataset_type}")

        for csv_file in csv_files[:10]:  # Limit to first 10 files for demo
            df = pd.read_csv(csv_file)
            all_data.append(df)

        if not all_data:
            raise ValueError(f"No CSV files found in {data_path}")

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined_data)} total samples")

        return combined_data

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
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=class_weight_dict,
            random_state=42
        )

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
            rf.fit(X_train_scaled, y_train)
            y_pred = rf.predict(X_val_scaled)

            if len(np.unique(y_val)) > 1:  # Only calculate F1 if both classes present
                f1 = f1_score(y_val, y_pred)
                scores.append(f1)

        if scores:
            avg_f1 = np.mean(scores)
            print(f"{event_name} - Average F1 Score: {avg_f1:.3f}")

        # Train final model on all data
        X_scaled = self.scaler.fit_transform(X)
        rf.fit(X_scaled, y)

        return rf

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

        # Evaluate each event type
        for event_col in event_cols:
            event_name = event_col.replace('event_', '')

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

    def analyze_configuration_impact(self, dataset_type: str = "features_generated"):
        """Analyze how different configurations affect flight patterns and events."""
        from code.data_generator.rocket_data_generator import RocketDataGenerator

        data_path = self.data_dir / dataset_type
        csv_files = list(data_path.glob("*.csv"))

        configs_and_events = []

        print(f"Analyzing configuration impact from {len(csv_files)} files...")

        for csv_file in csv_files[:20]:  # Limit for demo
            # Parse configuration from filename
            config = RocketDataGenerator.parse_config_from_filename(csv_file.name)

            # Load data and check for events
            df = pd.read_csv(csv_file)
            event_cols = [col for col in df.columns if col.startswith('event_')]

            # Count events in this simulation
            event_counts = {}
            for event_col in event_cols:
                event_counts[event_col] = df[event_col].sum()

            # Add to analysis
            config_entry = config.copy()
            config_entry['filename'] = csv_file.name
            config_entry.update(event_counts)
            configs_and_events.append(config_entry)

        # Convert to DataFrame for analysis
        analysis_df = pd.DataFrame(configs_and_events)

        print("\nConfiguration Impact Analysis:")
        print("=" * 50)

        # Show correlation between wind speed and event detection
        if 'wind_speed' in analysis_df.columns and len(analysis_df) > 5:
            print(f"Wind speed range: {analysis_df['wind_speed'].min():.1f} - {analysis_df['wind_speed'].max():.1f} m/s")

            for event_col in event_cols[:3]:  # Show first 3 event types
                if event_col in analysis_df.columns:
                    correlation = analysis_df['wind_speed'].corr(analysis_df[event_col])
                    print(f"Wind speed vs {event_col}: correlation = {correlation:.3f}")

        # Show temperature impact
        if 'temperature' in analysis_df.columns:
            print(f"\nTemperature range: {analysis_df['temperature'].min():.1f} - {analysis_df['temperature'].max():.1f} °C")

        # Show configuration with most/least events
        analysis_df['total_events'] = analysis_df[event_cols].sum(axis=1)

        if len(analysis_df) > 0:
            max_events_idx = analysis_df['total_events'].idxmax()
            min_events_idx = analysis_df['total_events'].idxmin()

            print(f"\nConfiguration with most events ({analysis_df.loc[max_events_idx, 'total_events']:.0f}):")
            max_config = analysis_df.loc[max_events_idx]
            for key in ['wind_speed', 'temperature', 'launch_angle']:
                if key in max_config:
                    print(f"  {key}: {max_config[key]}")

            print(f"\nConfiguration with least events ({analysis_df.loc[min_events_idx, 'total_events']:.0f}):")
            min_config = analysis_df.loc[min_events_idx]
            for key in ['wind_speed', 'temperature', 'launch_angle']:
                if key in min_config:
                    print(f"  {key}: {min_config[key]}")

        return analysis_df

def main():
    """Example usage of the EventClassifier."""

    print("Rocket Flight Event Classification Example")
    print("=" * 50)

    # Initialize classifier
    classifier = EventClassifier("./simulation_data")

    # Train classifiers on noisy feature-rich data
    try:
        classifier.train_all_classifiers("features_generated")

        # Evaluate on clean data to see how well it generalizes
        classifier.evaluate_model("features_ground_truth")

        # Plot feature importance for apogee detection
        classifier.plot_feature_importance("apogee")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo use this script:")
        print("1. Run the rocket data generator with include_event_labels=True and include_features=True")
        print("2. Make sure the simulation_data directory contains the generated datasets")
        print("\nExample:")
        print("python rocket_data_generator.py")


if __name__ == "__main__":
    main()

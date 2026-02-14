# Event Classification Guide for Rocket Flight Data

## Overview
This guide provides recommendations for training classifiers to detect events in rocket flight time series data using the generated datasets.

## Dataset Structure

The code now generates 4 types of datasets:

### 1. Basic Datasets
- **ground_truth/**: Clean simulation data without sensor noise
- **generated/**: Data with realistic sensor noise applied

### 2. Event-Labeled Datasets  
- **event_labeled_ground_truth/**: Clean data with event labels
- **event_labeled_generated/**: Noisy data with event labels

### 3. Feature-Rich Datasets
- **features_ground_truth/**: Clean data with engineered features
- **features_generated/**: Noisy data with engineered features

## Event Types Detected

The system labels these rocket flight events:
- **liftoff**: Rocket ignition and launch
- **burnout**: Motor burnout
- **apogee**: Maximum altitude reached
- **recovery_deployment**: Parachute/recovery system deployment
- **landing**: Ground contact
- **stage_separation**: Multi-stage rocket separation

## Labeling Strategy

### Binary Labels
- `event_{type}`: 1 if event is occurring, 0 otherwise
- `event_{type}_upcoming`: 1 if event will occur in next 5 seconds, 0 otherwise

### Time Windows
- Events are labeled with a configurable time window (default: 1 second)
- This accounts for the fact that events happen over brief periods

## Feature Engineering

### Rolling Window Features
- **Rolling statistics**: mean, std, min, max over 10-step windows
- **Derivatives**: First and second derivatives of all sensor signals
- **Rate of change**: Percentage change between time steps

### Cross-Sensor Features
- **Altitude-velocity ratio**: Relationship between position and velocity
- **Acceleration-velocity product**: Combined motion characteristics

## Recommended Classification Approaches

### 1. Traditional Machine Learning

#### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# For multi-event classification
rf = MultiOutputClassifier(RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced'  # Handle class imbalance
))
```

#### Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# Handle class imbalance with sample weights
class_weights = compute_sample_weight('balanced', y_train)
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6
)
gb.fit(X_train, y_train, sample_weight=class_weights)
```

### 2. Deep Learning Approaches

#### LSTM for Sequential Data
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(timesteps, n_features)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(n_event_types, activation='sigmoid')  # Multi-label classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)
```

#### 1D CNN for Pattern Detection
```python
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(timesteps, n_features)),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(50, activation='relu'),
    Dense(n_event_types, activation='sigmoid')
])
```

### 3. Time Series Specific Models

#### Prophet for Anomaly Detection
```python
from prophet import Prophet

# Use for detecting unusual patterns that might indicate events
# Combine with other classifiers for improved performance
```

## Data Preprocessing Recommendations

### 1. Handle Class Imbalance
Events are rare compared to normal flight data:
```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
```

### 2. Feature Scaling
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 3. Time Series Cross-Validation
```python
from sklearn.model_selection import TimeSeriesSplit

# Use time-aware cross-validation
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    # Train and validate
    pass
```

## Evaluation Strategies

### 1. Metrics for Imbalanced Data
```python
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Use F1-score, precision, recall instead of just accuracy
f1 = f1_score(y_true, y_pred, average='weighted')
```

### 2. Event-Specific Evaluation
```python
# Evaluate each event type separately
for event_type in event_types:
    event_idx = event_columns.index(f'event_{event_type}')
    print(f"{event_type}: {classification_report(y_true[:, event_idx], y_pred[:, event_idx])}")
```

### 3. Temporal Accuracy
- **Early Detection**: How far in advance can you predict events?
- **False Positive Rate**: Critical for safety-critical applications
- **Event Duration Accuracy**: How well do you capture the full event window?

## Advanced Techniques

### 1. Ensemble Methods
Combine multiple approaches:
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier([
    ('rf', RandomForestClassifier()),
    ('gb', GradientBoostingClassifier()),
    ('svm', SVC(probability=True))
], voting='soft')
```

### 2. Sequence-to-Sequence Models
For predicting upcoming events:
```python
# Use encoder-decoder LSTM to predict future event probabilities
# Input: Current and past sensor data
# Output: Event probabilities for next N time steps
```

### 3. Attention Mechanisms
```python
from tensorflow.keras.layers import Attention

# Add attention to focus on important time steps
# Useful for understanding which sensor patterns indicate specific events
```

## Implementation Tips

### 1. Start Simple
1. Begin with basic Random Forest on the feature-rich dataset
2. Use single-event binary classification before multi-event
3. Gradually add complexity

### 2. Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select most informative features
selector = SelectKBest(f_classif, k=50)
X_selected = selector.fit_transform(X_train, y_train)
```

### 3. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='f1_weighted'
)
```

## Dataset Usage Recommendations

1. **Training**: Use `features_generated` (includes noise and features)
2. **Validation**: Use `features_ground_truth` for clean evaluation
3. **Testing**: Mix of both to evaluate robustness to sensor noise
4. **Event Analysis**: Use `event_labeled_*` to understand event patterns

## Next Steps

1. Run the data generator with `include_event_labels=True` and `include_features=True`
2. Explore the generated datasets to understand event distribution
3. Start with Random Forest on the features dataset
4. Gradually move to more sophisticated approaches based on performance needs

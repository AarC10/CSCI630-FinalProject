# Rocket Flight Phase Classification

This project is meant for classifying high-power rocket flight phases from
multivariate time-series telemetry. The project trains models
on OpenRocket-generated sensor data and evaluates how well those models can identify
key flight events in both simulated and real flight data. 

Using OpenRocket-generated simulations, we train and compare logistic regression, k-nearest
neighbors, time-series forest, and MiniRocket classifiers on sliding windows of pressure, temperature, and acceleration data. The models are expected to perform well on simulated test data where the sensor distribution matches training, but real flight logs may show lower accuracy. The goal is to identify key events such as liftoff, burnout, apogee, and
recovery deployment while evaluating which model families offer the best tradeoff between accuracy, inference speed, and robustness.

## Team

- Aaron Chan
- Shikher Shah
- Chinmay Lokare

## Overview

The classifier uses sliding windows over four sensor channels:

- `AIR_PRESSURE`
- `AIR_TEMPERATURE`
- `ACCELERATION_XY`
- `ACCELERATION_Z`

Each window receives the majority `flight_phase` label inside the window. 
The following phases are

| Label | Phase |
| --- | --- |
| 0 | `no_event` |
| 1 | `liftoff` |
| 2 | `burnout` |
| 3 | `apogee` |
| 4 | `recovery_deployment` |

The default window size is 100 timesteps with a stride of 50 timesteps. For the
100 Hz datasets used here, this means 1 second windows advanced every
0.5 seconds.

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

OpenRocket-based data generation also requires a local OpenRocket JAR. The
helper scripts expect it at:

```text
~/OpenRocket/OpenRocket.jar
```

Training and classification against already-generated CSV files do not require
OpenRocket.

## Running

### Training

Run training through `code/main.py train`:

```bash
python3 code/main.py train \
  --input-dir simulation_data \
  --output-dir outputs \
  --model minirocket \
  --model-kwargs num_kernels=1000 solver=saga max_iter=3000 tol=1e-4 \
  --sequence-length 100 \
  --stride 50
```


Supported model types are:

- `lr`
- `knn`
- `tsf`
- `minirocket`
- `catch22_lr`

Useful options:

```bash
--max-files 100                 # train on only the first 100 CSV files
--test-size 0.2                 # holdout fraction
--curve-fractions 0.2 0.5 1.0   # training fractions for learning curves
--no-stratify                   # disable stratified train/test split
--random-state 42
```

Each run writes a timestamped directory under `outputs/`, for example:

```text
outputs/20260326_115022_train_minirocket/
```

Training artifacts include:

- `config.json`
- `metrics.json`
- `dataset_summary.json`
- `learning_curves.json`
- `test_predictions.csv`
- `model.pkl`
- `plots/`

The training command also promotes the best model by test weighted F1 to:

```text
outputs/best_model.pkl
outputs/best_model_metadata.json
```

There are also several examples in the shell scripts at the repo's top level

### Classifying One CSV

Classify a labeled CSV file with a saved model:

```bash
python3 code/main.py classify \
  --model-path outputs/best_model.pkl \
  --input-file FLIGHT.csv \
  --output-dir outputs
```

`--model-path` may point to a `model.pkl` file or to a run directory that
contains `model.pkl`. If `--sequence-length` and `--stride` are not set, the CLI
uses values saved in the model run config if aailable

Classification artifacts include:

- `config.json`
- `metrics.json`
- `predictions.csv`
- `plots/confusion_matrix.png`
- `plots/prediction_timeline.png`
- `plots/predicted_label_distribution.png`
- `plots/actual_label_distribution.png`

### Data Generation

The main OpenRocket generator is in:

```text
code/data_generator/rocket_data_generator.py
```

It runs a grid of launch conditions across `.ork` files and writes labeled CSVs.

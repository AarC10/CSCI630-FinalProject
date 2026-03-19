from __future__ import annotations


import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Dataloader:
    SENSOR_COLS = ["AIR_PRESSURE", "AIR_TEMPERATURE", "ACCELERATION_XY", "ACCELERATION_Z"]
    LABEL_COL = "flight_phase"
    TIME_COL = "TIME"

    # Flight phase mapping
    PHASE_MAPPING = {
        0: 'no_event',
        1: 'liftoff',
        2: 'burnout',
        3: 'apogee',
        4: 'recovery_deployment',
        5: 'landing'
    }

    def __init__(self, data_path: str, sequence_length: int = 100, stride: int = 50,
                 test_size: float = 0.2, random_state: int = 42, progress_every_files: int = 100):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.stride = stride
        self.test_size = test_size
        self.random_state = random_state
        self.progress_every_files = max(1, int(progress_every_files))

        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")

        logging.info(f"DataLoader initialized with sequence_length={sequence_length}, "
                    f"stride={stride}, test_size={test_size}, progress_every_files={self.progress_every_files}")

    def _validate_required_columns(self, df: pd.DataFrame, source_name: str) -> None:
        missing_cols = set(self.SENSOR_COLS + [self.LABEL_COL]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"{source_name} is missing required columns: {sorted(missing_cols)}")

    def _create_sliding_windows_with_metadata(self, df: pd.DataFrame, source_name: str = "unknown") -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        sensor_data = df[self.SENSOR_COLS].values
        labels = df[self.LABEL_COL].values
        times = df[self.TIME_COL].values if self.TIME_COL in df.columns else None

        n_timesteps, n_channels = sensor_data.shape
        n_windows = (n_timesteps - self.sequence_length) // self.stride + 1

        if n_windows <= 0:
            return np.array([]), np.array([]), []

        X_windows = []
        y_windows = []
        metadata = []

        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.sequence_length

            window = sensor_data[start_idx:end_idx]
            window_labels = labels[start_idx:end_idx]

            if np.any(np.isnan(window)):
                continue

            unique, counts = np.unique(window_labels, return_counts=True)
            majority_label = unique[np.argmax(counts)]

            X_windows.append(window.T)
            y_windows.append(majority_label)
            metadata.append(
                {
                    "source_file": source_name,
                    "window_index": len(metadata),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx - 1),
                    "start_time": None if times is None else float(times[start_idx]),
                    "end_time": None if times is None else float(times[end_idx - 1]),
                    "majority_label": int(majority_label),
                    "majority_label_name": self.PHASE_MAPPING.get(int(majority_label), "unknown"),
                }
            )

        if len(X_windows) == 0:
            return np.array([]), np.array([]), []

        return np.array(X_windows), np.array(y_windows), metadata

    def _create_sliding_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_windows, y_windows, _ = self._create_sliding_windows_with_metadata(df)
        return X_windows, y_windows

    def load_file(self, file_path: str | Path, include_metadata: bool = False):
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"CSV file does not exist: {file_path}")

        df = pd.read_csv(file_path)
        self._validate_required_columns(df, file_path.name)
        X_windows, y_windows, metadata = self._create_sliding_windows_with_metadata(df, source_name=file_path.name)

        if len(X_windows) == 0:
            raise ValueError(f"No valid windows could be generated from {file_path}")

        return (X_windows, y_windows, metadata) if include_metadata else (X_windows, y_windows)

    def load_data(self, max_files: int = None) -> Tuple[np.ndarray, np.ndarray]:
        csv_files = sorted(list(self.data_path.glob("*.csv")))

        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in {self.data_path}")

        if max_files is not None:
            csv_files = csv_files[:max_files]

        logging.info(f"Loading {len(csv_files)} CSV files from {self.data_path}")

        all_X = []
        all_y = []
        skipped_files = 0

        for i, file_path in enumerate(csv_files):
            try:
                df = pd.read_csv(file_path)
                self._validate_required_columns(df, file_path.name)

                X_windows, y_windows = self._create_sliding_windows(df)

                if len(X_windows) > 0:
                    all_X.append(X_windows)
                    all_y.append(y_windows)

                if (i + 1) % self.progress_every_files == 0:
                    logging.info(
                        f"Processed {i + 1}/{len(csv_files)} files "
                        f"({(i + 1) / len(csv_files):.1%})"
                    )

            except Exception as e:
                logging.warning(f"Skipping {file_path.name}: {e}")
                skipped_files += 1
                continue

        if len(all_X) == 0:
            raise ValueError("No valid data could be loaded")

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        logging.info(f"Loaded {len(csv_files) - skipped_files} files successfully "
                    f"({skipped_files} skipped)")
        logging.info(f"Generated {len(X)} windows with shape {X.shape}")
        logging.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        return X, y

    def get_train_test_split(self, max_files: int = None,
                           stratify: bool = True) -> Tuple[np.ndarray, np.ndarray,
                                                            np.ndarray, np.ndarray]:
        X, y = self.load_data(max_files=max_files)

        stratify_by = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_by
        )

        logging.info(f"Train set: {len(X_train)} samples")
        logging.info(f"Test set: {len(X_test)} samples")
        logging.info(f"Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        logging.info(f"Test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

        return X_train, X_test, y_train, y_test

    def estimate_memory(self, n_files: int = None) -> dict:
        csv_files = list(self.data_path.glob("*.csv"))

        if n_files is None:
            n_files = len(csv_files)
        else:
            n_files = min(n_files, len(csv_files))

        if n_files == 0:
            return {"error": "No CSV files found"}

        sample_file = csv_files[0]
        df = pd.read_csv(sample_file)
        self._validate_required_columns(df, sample_file.name)

        n_timesteps = len(df)
        n_channels = len(self.SENSOR_COLS)

        # Estimate windows per file
        n_windows_per_file = max(0, (n_timesteps - self.sequence_length) // self.stride + 1)

        # Total windows
        total_windows = n_windows_per_file * n_files
        X_memory_bytes = total_windows * n_channels * self.sequence_length * 8
        y_memory_bytes = total_windows * 8
        total_memory_bytes = X_memory_bytes + y_memory_bytes

        return {
            "n_files": n_files,
            "estimated_windows_per_file": n_windows_per_file,
            "total_estimated_windows": total_windows,
            "X_memory_MB": X_memory_bytes / (1024 ** 2),
            "y_memory_MB": y_memory_bytes / (1024 ** 2),
            "total_memory_MB": total_memory_bytes / (1024 ** 2),
            "total_memory_GB": total_memory_bytes / (1024 ** 3),
            "X_shape": (total_windows, n_channels, self.sequence_length),
            "y_shape": (total_windows,)
        }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataloader for rocket sensor data")
    parser.add_argument("--data_path", type=str, required=True, help="Path to directory containing CSV files")

    args = parser.parse_args()
    dataloader = Dataloader(args.data_path)
    memory_info = dataloader.estimate_memory(n_files=1000)
    logging.info(f"Memory estimation for 1000 files: {memory_info}")
    X_train, X_test, y_train, y_test = dataloader.get_train_test_split()

    logging.info(f"Final train set shape: {X_train.shape}, {y_train.shape}")
    logging.info(f"Final test set shape: {X_test.shape}, {y_test.shape}")
    logging.info(f"Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    logging.info(f"Test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")


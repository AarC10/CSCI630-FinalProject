from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Dataloader:
    SENSOR_COLS = ["AIR_PRESSURE", "AIR_TEMPERATURE", "ACCELERATION_XY", "ACCELERATION_Z"]
    LABEL_COL = "flight_phase"
    TIME_COL = "TIME"

    PHASE_MAPPING = {
        0: "no_event",
        1: "liftoff",
        2: "burnout",
        3: "apogee",
        4: "recovery_deployment",
        5: "landing",
    }

    SUPPORTED_LABEL_STRATEGIES = {"majority", "center", "trailing"}

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 100,
        stride: int = 50,
        test_size: float = 0.2,
        random_state: int = 42,
        progress_every_files: int = 100,
        label_strategy: str = "center",
        min_label_purity: float = 0.0,
        split_strategy: str = "group",
    ):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.stride = stride
        self.test_size = test_size
        self.random_state = random_state
        self.progress_every_files = max(1, int(progress_every_files))
        self.label_strategy = label_strategy
        self.min_label_purity = float(min_label_purity)
        self.split_strategy = split_strategy

        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        if self.label_strategy not in self.SUPPORTED_LABEL_STRATEGIES:
            raise ValueError(
                f"Unsupported label_strategy '{self.label_strategy}'. "
                f"Expected one of {sorted(self.SUPPORTED_LABEL_STRATEGIES)}."
            )
        if not 0.0 <= self.min_label_purity <= 1.0:
            raise ValueError("min_label_purity must be in [0.0, 1.0].")
        if self.split_strategy not in {"window", "group"}:
            raise ValueError("split_strategy must be either 'window' or 'group'.")

        logging.info(
            "DataLoader initialized with sequence_length=%s, stride=%s, test_size=%s, "
            "label_strategy=%s, min_label_purity=%.2f, split_strategy=%s, progress_every_files=%s",
            sequence_length,
            stride,
            test_size,
            self.label_strategy,
            self.min_label_purity,
            self.split_strategy,
            self.progress_every_files,
        )

    def _validate_required_columns(self, df: pd.DataFrame, source_name: str) -> None:
        missing_cols = set(self.SENSOR_COLS + [self.LABEL_COL]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"{source_name} is missing required columns: {sorted(missing_cols)}")

    def _resolve_window_label(self, window_labels: np.ndarray) -> tuple[int, float]:
        unique, counts = np.unique(window_labels, return_counts=True)
        majority_index = int(np.argmax(counts))
        majority_label = int(unique[majority_index])
        majority_purity = float(counts[majority_index] / len(window_labels))

        if self.label_strategy == "majority":
            return majority_label, majority_purity
        if self.label_strategy == "center":
            center_index = len(window_labels) // 2
            center_label = int(window_labels[center_index])
            center_purity = float(np.mean(window_labels == center_label))
            return center_label, center_purity
        if self.label_strategy == "trailing":
            trailing_label = int(window_labels[-1])
            trailing_purity = float(np.mean(window_labels == trailing_label))
            return trailing_label, trailing_purity
        raise RuntimeError(f"Unhandled label_strategy: {self.label_strategy}")

    def _create_sliding_windows_with_metadata(
        self,
        df: pd.DataFrame,
        source_name: str = "unknown",
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        sensor_data = df[self.SENSOR_COLS].values
        labels = df[self.LABEL_COL].values
        times = df[self.TIME_COL].values if self.TIME_COL in df.columns else None

        n_timesteps, _ = sensor_data.shape
        n_windows = (n_timesteps - self.sequence_length) // self.stride + 1
        if n_windows <= 0:
            return np.array([]), np.array([]), []

        X_windows: list[np.ndarray] = []
        y_windows: list[int] = []
        metadata: list[Dict[str, Any]] = []

        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.sequence_length
            window = sensor_data[start_idx:end_idx]
            window_labels = labels[start_idx:end_idx]

            if np.any(np.isnan(window)):
                continue

            selected_label, label_purity = self._resolve_window_label(window_labels)
            if label_purity < self.min_label_purity:
                continue

            center_idx = start_idx + (self.sequence_length // 2)
            center_label = int(labels[min(center_idx, len(labels) - 1)])

            X_windows.append(window.T.astype(np.float32, copy=False))
            y_windows.append(selected_label)
            metadata.append(
                {
                    "source_file": source_name,
                    "group": source_name,
                    "window_index": len(metadata),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx - 1),
                    "start_time": None if times is None else float(times[start_idx]),
                    "end_time": None if times is None else float(times[end_idx - 1]),
                    "selected_label": int(selected_label),
                    "selected_label_name": self.PHASE_MAPPING.get(int(selected_label), "unknown"),
                    "center_label": center_label,
                    "center_label_name": self.PHASE_MAPPING.get(center_label, "unknown"),
                    "label_purity": float(label_purity),
                }
            )

        if not X_windows:
            return np.array([]), np.array([]), []

        return np.stack(X_windows), np.asarray(y_windows, dtype=np.int64), metadata

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

    def load_data(
        self,
        max_files: int = None,
        include_metadata: bool = False,
        include_groups: bool = False,
    ):
        csv_files = sorted(self.data_path.glob("*.csv"))
        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in {self.data_path}")

        if max_files is not None:
            csv_files = csv_files[:max_files]

        logging.info(f"Loading {len(csv_files)} CSV files from {self.data_path}")

        all_X: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        all_groups: list[np.ndarray] = []
        all_metadata: list[Dict[str, Any]] = []
        skipped_files = 0

        for i, file_path in enumerate(csv_files):
            try:
                df = pd.read_csv(file_path)
                self._validate_required_columns(df, file_path.name)
                X_windows, y_windows, metadata = self._create_sliding_windows_with_metadata(
                    df,
                    source_name=file_path.name,
                )

                if len(X_windows) > 0:
                    all_X.append(X_windows)
                    all_y.append(y_windows)
                    all_groups.append(np.full(len(y_windows), file_path.name, dtype=object))
                    if include_metadata:
                        all_metadata.extend(metadata)

                if (i + 1) % self.progress_every_files == 0:
                    logging.info(
                        "Processed %s/%s files (%.1f%%)",
                        i + 1,
                        len(csv_files),
                        (i + 1) / len(csv_files) * 100,
                    )
            except Exception as exc:
                logging.warning(f"Skipping {file_path.name}: {exc}")
                skipped_files += 1

        if not all_X:
            raise ValueError("No valid data could be loaded")

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        groups = np.concatenate(all_groups, axis=0)

        logging.info("Loaded %s files successfully (%s skipped)", len(csv_files) - skipped_files, skipped_files)
        logging.info("Generated %s windows with shape %s", len(X), X.shape)
        logging.info("Class distribution: %s", dict(zip(*np.unique(y, return_counts=True))))

        outputs: list[Any] = [X, y]
        if include_groups:
            outputs.append(groups)
        if include_metadata:
            outputs.append(all_metadata)
        if len(outputs) == 2:
            return X, y
        return tuple(outputs)

    def _slice_split_outputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        groups: np.ndarray | None = None,
        metadata: list[Dict[str, Any]] | None = None,
        include_metadata: bool = False,
    ):
        train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))
        test_idx = np.sort(np.asarray(test_idx, dtype=np.int64))

        outputs: list[Any] = [X[train_idx], X[test_idx], y[train_idx], y[test_idx]]

        if groups is not None:
            outputs.extend([groups[train_idx], groups[test_idx]])
        if include_metadata:
            metadata = metadata or []
            outputs.extend(
                [
                    [metadata[index] for index in train_idx],
                    [metadata[index] for index in test_idx],
                ]
            )
        return tuple(outputs)

    def _group_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        metadata: list[Dict[str, Any]] | None,
        include_groups: bool,
        include_metadata: bool,
        stratify: bool,
    ):
        if len(np.unique(groups)) < 2:
            raise ValueError("Grouped splitting requires at least two source files.")

        if stratify:
            try:
                n_splits = max(2, int(round(1.0 / self.test_size)))
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
                train_idx, test_idx = next(splitter.split(X, y, groups))
            except Exception:
                splitter = GroupShuffleSplit(
                    n_splits=1,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
                train_idx, test_idx = next(splitter.split(X, y, groups))
        else:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            train_idx, test_idx = next(splitter.split(X, y, groups))

        return self._slice_split_outputs(
            X,
            y,
            train_idx,
            test_idx,
            groups=groups if include_groups else None,
            metadata=metadata,
            include_metadata=include_metadata,
        )

    def get_train_test_split(
        self,
        max_files: int = None,
        stratify: bool = True,
        include_metadata: bool = False,
        include_groups: bool = False,
    ):
        X, y, groups, metadata = self.load_data(
            max_files=max_files,
            include_metadata=True,
            include_groups=True,
        )

        if self.split_strategy == "group":
            split_outputs = self._group_train_test_split(
                X=X,
                y=y,
                groups=groups,
                metadata=metadata,
                include_groups=include_groups,
                include_metadata=include_metadata,
                stratify=stratify,
            )
        else:
            stratify_by = y if stratify else None
            train_idx, test_idx = train_test_split(
                np.arange(len(X)),
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify_by,
            )
            split_outputs = self._slice_split_outputs(
                X,
                y,
                train_idx,
                test_idx,
                groups=groups if include_groups else None,
                metadata=metadata,
                include_metadata=include_metadata,
            )

        X_train, X_test, y_train, y_test, *extras = split_outputs

        logging.info("Train set: %s samples", len(X_train))
        logging.info("Test set: %s samples", len(X_test))
        logging.info("Train class distribution: %s", dict(zip(*np.unique(y_train, return_counts=True))))
        logging.info("Test class distribution: %s", dict(zip(*np.unique(y_test, return_counts=True))))

        if self.split_strategy == "group" and include_groups:
            train_groups = extras[0]
            test_groups = extras[1]
            logging.info("Train groups: %s unique source files", len(np.unique(train_groups)))
            logging.info("Test groups: %s unique source files", len(np.unique(test_groups)))

        if not include_groups and not include_metadata:
            return X_train, X_test, y_train, y_test
        return split_outputs

    def iter_group_folds(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        n_splits: int,
    ):
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            iterator = splitter.split(X, y, groups)
        except Exception:
            splitter = GroupShuffleSplit(
                n_splits=n_splits,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            iterator = splitter.split(X, y, groups)

        for fold_index, (train_idx, test_idx) in enumerate(iterator, start=1):
            yield fold_index, np.sort(train_idx), np.sort(test_idx)

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
        n_windows_per_file = max(0, (n_timesteps - self.sequence_length) // self.stride + 1)

        total_windows = n_windows_per_file * n_files
        bytes_per_value = np.dtype(np.float32).itemsize
        X_memory_bytes = total_windows * n_channels * self.sequence_length * bytes_per_value
        y_memory_bytes = total_windows * np.dtype(np.int64).itemsize
        total_memory_bytes = X_memory_bytes + y_memory_bytes

        return {
            "n_files": n_files,
            "estimated_windows_per_file": n_windows_per_file,
            "total_estimated_windows": total_windows,
            "X_memory_MB": X_memory_bytes / (1024 ** 2),
            "y_memory_MB": y_memory_bytes / (1024 ** 2),
            "total_memory_MB": total_memory_bytes / (1024 ** 2),
            "total_memory_GB": total_memory_bytes / (1024 ** 3),
            "X_dtype": "float32",
            "y_dtype": "int64",
            "X_shape": (total_windows, n_channels, self.sequence_length),
            "y_shape": (total_windows,),
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

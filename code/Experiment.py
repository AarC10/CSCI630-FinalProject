from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd


class Experiment:
    @staticmethod
    def _json_default(value: Any):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, tuple):
            return list(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def __init__(self, output_root: str | Path, action: str, model_name: str):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{timestamp}_{action}_{model_name}"
        self.run_dir = self.output_root / base_name
        suffix = 1
        while self.run_dir.exists():
            self.run_dir = self.output_root / f"{base_name}_{suffix}"
            suffix += 1
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def path(self, relative_path: str) -> Path:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, relative_path: str, payload: Mapping[str, Any]) -> Path:
        path = self.path(relative_path)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=self._json_default)
        return path

    def save_text(self, relative_path: str, text: str) -> Path:
        path = self.path(relative_path)
        path.write_text(text, encoding="utf-8")
        return path

    def save_pickle(self, relative_path: str, payload: Any) -> Path:
        path = self.path(relative_path)
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        return path

    def save_dataframe(self, relative_path: str, dataframe: pd.DataFrame) -> Path:
        path = self.path(relative_path)
        dataframe.to_csv(path, index=False)
        return path

    @staticmethod
    def load_pickle(path: str | Path) -> Any:
        with Path(path).open("rb") as handle:
            return pickle.load(handle)

    def plot_confusion_matrix(
        self,
        y_true: Iterable[int],
        y_pred: Iterable[int],
        labels: list[int],
        label_names: list[str],
        relative_path: str,
        title: str,
    ) -> Path:
        pass

    def plot_metric_curve(
        self,
        x_values: Iterable[float],
        train_values: Iterable[float],
        eval_values: Iterable[float],
        relative_path: str,
        title: str,
        ylabel: str,
        xlabel: str = "Training fraction",
    ) -> Path:
        pass

    def plot_label_distribution(
        self,
        y: Iterable[int],
        phase_mapping: Mapping[int, str],
        relative_path: str,
        title: str,
    ) -> Path:
        pass

    def plot_phase_timeline(
        self,
        x_axis: Iterable[float],
        predictions: Iterable[int],
        phase_mapping: Mapping[int, str],
        relative_path: str,
        title: str,
        ground_truth: Iterable[int] | None = None,
        comparison_predictions: Iterable[int] | None = None,
        comparison_label: str = "comparison",
        xlabel: str = "Window index",
    ) -> Path:
        pass


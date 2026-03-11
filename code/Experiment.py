from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import pandas as pd


class Experiment:
    @staticmethod
    def _json_default(value: Any):
        pass

    def __init__(self, output_root: str | Path, action: str, model_name: str):
        pass

    def path(self, relative_path: str) -> Path:
        pass

    def save_json(self, relative_path: str, payload: Mapping[str, Any]) -> Path:
        pass

    def save_text(self, relative_path: str, text: str) -> Path:
        pass

    def save_pickle(self, relative_path: str, payload: Any) -> Path:
        pass

    def save_dataframe(self, relative_path: str, dataframe: pd.DataFrame) -> Path:
        pass

    @staticmethod
    def load_pickle(path: str | Path) -> Any:
        pass

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


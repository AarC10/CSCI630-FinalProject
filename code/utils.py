from __future__ import annotations

import argparse
import ast
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Experiment import Experiment
from PhaseClassifier import FlightPhaseClassifier

PHASE_MAPPING = FlightPhaseClassifier.PHASE_MAPPING
PHASE_LABELS = sorted(PHASE_MAPPING)
PHASE_NAMES = [PHASE_MAPPING[label] for label in PHASE_LABELS]
DEFAULT_CURVE_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

def _coerce_cli_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_model_kwargs(entries: list[str] | None) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Invalid model hyperparameter '{entry}'. Expected KEY=VALUE.")
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid model hyperparameter '{entry}'. Key cannot be empty.")
        model_kwargs[key] = _coerce_cli_value(raw_value.strip())
    return model_kwargs


def resolve_model_path(model_path: str | Path) -> tuple[Path, Path]:
    candidate = Path(model_path).expanduser().resolve()
    if candidate.is_dir():
        resolved_model = candidate / "model.pkl"
        if not resolved_model.exists():
            raise FileNotFoundError(f"Could not find model.pkl in {candidate}")
        return candidate, resolved_model
    if candidate.is_file():
        return candidate.parent, candidate
    raise FileNotFoundError(f"Model path does not exist: {candidate}")


def load_saved_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_window_params(args: argparse.Namespace, saved_config: dict[str, Any]) -> tuple[int, int]:
    sequence_length = args.sequence_length or saved_config.get("sequence_length") or 100
    stride = args.stride or saved_config.get("stride") or 50
    return int(sequence_length), int(stride)


def select_x_axis(metadata: list[dict[str, Any]]) -> tuple[np.ndarray, str]:
    if metadata and all(item.get("start_time") is not None for item in metadata):
        return np.asarray([item["start_time"] for item in metadata], dtype=float), "Time"
    return np.asarray([item["window_index"] for item in metadata], dtype=float), "Window index"


def phase_name(label: int) -> str:
    return PHASE_MAPPING.get(int(label), str(label))


def build_predictions_dataframe(
    metadata: list[dict[str, Any]],
    predictions: np.ndarray,
    ground_truth: np.ndarray | None = None,
) -> pd.DataFrame:
    dataframe = pd.DataFrame(metadata)
    if dataframe.empty:
        dataframe = pd.DataFrame({"window_index": np.arange(len(predictions), dtype=int)})
    dataframe["predicted_phase"] = predictions.astype(int)
    dataframe["predicted_phase_name"] = [phase_name(label) for label in predictions]
    if ground_truth is not None:
        dataframe["actual_phase"] = ground_truth.astype(int)
        dataframe["actual_phase_name"] = [phase_name(label) for label in ground_truth]
        dataframe["is_correct"] = dataframe["predicted_phase"] == dataframe["actual_phase"]
    return dataframe


def load_model(model_path: str | Path) -> tuple[Any, Path, dict[str, Any]]:
    run_dir, resolved_model_path = resolve_model_path(model_path)
    classifier = Experiment.load_pickle(resolved_model_path)

    required_attrs = ("predict", "evaluate", "model_type")
    missing_attrs = [attribute for attribute in required_attrs if not hasattr(classifier, attribute)]
    if missing_attrs:
        raise TypeError(
            f"Saved model at {resolved_model_path} is missing required classifier attributes: {missing_attrs}."
        )

    return classifier, run_dir, load_saved_config(run_dir)

from __future__ import annotations

import argparse
import ast
import importlib
import json
import shutil
from datetime import datetime, timezone
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
BEST_MODEL_FILENAME = "best_model.pkl"
BEST_MODEL_METADATA_FILENAME = "best_model_metadata.json"

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


def smooth_predictions_by_group(
    predictions: np.ndarray,
    metadata: list[dict[str, Any]] | None = None,
    window_size: int = 1,
) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.int64)
    if window_size <= 1 or len(predictions) == 0:
        return predictions.copy()

    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer.")

    smoothed = predictions.copy()
    half_window = window_size // 2
    if metadata:
        groups = [str(item.get("source_file", "global")) for item in metadata]
    else:
        groups = ["global"] * len(predictions)

    start = 0
    while start < len(predictions):
        end = start + 1
        while end < len(predictions) and groups[end] == groups[start]:
            end += 1

        segment = predictions[start:end]
        for offset in range(len(segment)):
            left = max(0, offset - half_window)
            right = min(len(segment), offset + half_window + 1)
            window = segment[left:right]
            labels, counts = np.unique(window, return_counts=True)
            winning_label = labels[np.argmax(counts)]
            smoothed[start + offset] = int(winning_label)
        start = end

    return smoothed


def best_model_path(output_root: str | Path) -> Path:
    return Path(output_root).expanduser().resolve() / BEST_MODEL_FILENAME


def best_model_metadata_path(output_root: str | Path) -> Path:
    return Path(output_root).expanduser().resolve() / BEST_MODEL_METADATA_FILENAME


def load_best_model_metadata(output_root: str | Path) -> dict[str, Any] | None:
    metadata_path = best_model_metadata_path(output_root)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _atomic_copy_file(source_path: str | Path, destination_path: str | Path) -> Path:
    source = Path(source_path).expanduser().resolve()
    destination = Path(destination_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temp_path)
    temp_path.replace(destination)
    return destination


def _atomic_write_json(destination_path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(destination_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temp_path.replace(destination)
    return destination


def promote_best_model(
    output_root: str | Path,
    run_dir: str | Path,
    run_model_path: str | Path,
    score: float,
    metrics_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root_path = Path(output_root).expanduser().resolve()
    output_root_path.mkdir(parents=True, exist_ok=True)
    metadata_path = best_model_metadata_path(output_root_path)
    model_path = best_model_path(output_root_path)
    current_best = load_best_model_metadata(output_root_path)
    current_best_score = None if current_best is None else float(current_best.get("weighted_f1", float("-inf")))

    promoted = current_best is None or float(score) > current_best_score
    result = {
        "promoted": promoted,
        "selection_metric": "test.weighted_f1",
        "candidate_weighted_f1": float(score),
        "previous_best_weighted_f1": current_best_score,
        "best_model_path": str(model_path),
        "best_model_metadata_path": str(metadata_path),
    }

    if not promoted:
        result["active_source_run_dir"] = None if current_best is None else current_best.get("source_run_dir")
        return result

    copied_model_path = _atomic_copy_file(run_model_path, model_path)
    metadata = {
        "selection_metric": "test.weighted_f1",
        "weighted_f1": float(score),
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(Path(run_dir).expanduser().resolve()),
        "source_model_path": str(Path(run_model_path).expanduser().resolve()),
        "best_model_path": str(copied_model_path),
    }
    if metrics_path is not None:
        metadata["source_metrics_path"] = str(Path(metrics_path).expanduser().resolve())
    if config is not None:
        metadata.update(
            {
                "model": config.get("model"),
                "sequence_length": config.get("sequence_length"),
                "stride": config.get("stride"),
                "random_state": config.get("random_state"),
                "test_size": config.get("test_size"),
            }
        )
    _atomic_write_json(metadata_path, metadata)
    result["active_source_run_dir"] = metadata["source_run_dir"]
    return result


def load_model(model_path: str | Path) -> tuple[Any, Path, dict[str, Any]]:
    run_dir, resolved_model_path = resolve_model_path(model_path)
    classifier = Experiment.load_pickle(resolved_model_path)

    required_attrs = ("predict", "evaluate", "model_type")
    missing_attrs = [attribute for attribute in required_attrs if not hasattr(classifier, attribute)]
    if missing_attrs:
        raise TypeError(
            f"Saved model at {resolved_model_path} is missing required classifier attributes: {missing_attrs}."
        )

    saved_config = load_saved_config(run_dir)
    if not saved_config and resolved_model_path.name == BEST_MODEL_FILENAME:
        metadata = load_best_model_metadata(run_dir)
        source_run_dir = None if metadata is None else metadata.get("source_run_dir")
        if source_run_dir:
            saved_config = load_saved_config(Path(source_run_dir))

    return classifier, run_dir, saved_config

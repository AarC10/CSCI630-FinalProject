import argparse
import logging
import traceback

from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from sklearn.model_selection import train_test_split

from PhaseClassifier import FlightPhaseClassifier
from Dataloader import Dataloader
from Experiment import Experiment
from utils import *

PHASE_MAPPING = FlightPhaseClassifier.PHASE_MAPPING
PHASE_LABELS = sorted(PHASE_MAPPING)
PHASE_NAMES = [PHASE_MAPPING[label] for label in PHASE_LABELS]
DEFAULT_CURVE_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
LOGGER = logging.getLogger(__name__)


def run_classify(args: argparse.Namespace) -> int:
    classifier, run_dir, saved_config = load_model(args.model_path)
    sequence_length, stride = resolve_window_params(args, saved_config)
    input_file = Path(args.input_file).expanduser().resolve()
    smoothing_window = args.smoothing_window or saved_config.get("smoothing_window", 1)
    label_strategy = saved_config.get("label_strategy", args.label_strategy or "center")
    min_label_purity = saved_config.get("min_label_purity", args.min_label_purity)

    dataloader = Dataloader(
        data_path=str(input_file.parent),
        sequence_length=sequence_length,
        stride=stride,
        random_state=args.random_state,
        label_strategy=label_strategy,
        min_label_purity=min_label_purity,
        split_strategy=saved_config.get("split_strategy", "group"),
    )
    X, y, metadata = dataloader.load_file(input_file, include_metadata=True)

    predictions = classifier.predict(X)
    smoothed_predictions = smooth_predictions_by_group(predictions, metadata, window_size=smoothing_window)
    evaluation = classifier.score_predictions(y, smoothed_predictions)

    experiment = Experiment(args.output_dir, action="classify", model_name=classifier.model_type)
    experiment.save_json(
        "config.json",
        {
            "command": "classify",
            "model_path": resolve_model_path(args.model_path)[1],
            "source_run_dir": run_dir,
            "input_file": input_file,
            "sequence_length": sequence_length,
            "stride": stride,
            "label_strategy": label_strategy,
            "min_label_purity": min_label_purity,
            "smoothing_window": smoothing_window,
            "random_state": args.random_state,
        },
    )

    prediction_df = build_predictions_dataframe(metadata, smoothed_predictions, y)
    prediction_df["raw_predicted_phase"] = predictions.astype(int)
    prediction_df["raw_predicted_phase_name"] = [phase_name(label) for label in predictions]
    experiment.save_dataframe("predictions.csv", prediction_df)
    experiment.save_json("metrics.json", evaluation)
    experiment.plot_confusion_matrix(y, smoothed_predictions, PHASE_LABELS, PHASE_NAMES, "plots/confusion_matrix.png",
                                     f"Confusion matrix: {input_file.name}")
    x_axis, xlabel = select_x_axis(metadata)
    experiment.plot_phase_timeline(
        x_axis,
        smoothed_predictions,
        PHASE_MAPPING,
        "plots/prediction_timeline.png",
        f"Prediction timeline: {input_file.name}",
        ground_truth=y,
        xlabel=xlabel,
    )
    experiment.plot_label_distribution(smoothed_predictions, PHASE_MAPPING, "plots/predicted_label_distribution.png",
                                       "Predicted label distribution")
    experiment.plot_label_distribution(y, PHASE_MAPPING, "plots/actual_label_distribution.png",
                                       "Actual label distribution")

    print(f"Classification complete. saved to: {experiment.run_dir}")
    print(f"Accuracy: {evaluation['accuracy']:.4f} | Weighted F1: {evaluation['weighted_f1']:.4f}")
    return 0


def run_compare(args: argparse.Namespace) -> int:
    pass

def _extract_loss_curve(candidate: Any) -> list[float] | None:
    if candidate is None:
        return None

    visited: set[int] = set()
    queue = [candidate]
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))

        loss_curve = getattr(current, "loss_curve_", None)
        if loss_curve is not None:
            curve = [float(value) for value in loss_curve]
            if len(curve) > 1:
                return curve

        if hasattr(current, "named_steps"):
            queue.extend(current.named_steps.values())
        if hasattr(current, "steps"):
            queue.extend(step for _, step in current.steps)
        if hasattr(current, "model"):
            queue.append(current.model)

    return None


def _sample_training_subset(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples >= len(X):
        return X, y

    try:
        X_subset, _, y_subset, _ = train_test_split(
            X,
            y,
            train_size=n_samples,
            stratify=y,
            random_state=random_state,
        )
    except ValueError:
        X_subset, _, y_subset, _ = train_test_split(
            X,
            y,
            train_size=n_samples,
            random_state=random_state,
        )
    return X_subset, y_subset


def compute_cross_validation(
    model_type: str,
    model_kwargs: dict[str, Any],
    random_state: int | None,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    dataloader: Dataloader,
    n_splits: int,
    smoothing_window: int = 1,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    for fold_index, train_idx, test_idx in dataloader.iter_group_folds(X, y, groups, n_splits=n_splits):
        LOGGER.info(
            "Cross-validation fold %d/%d | train=%d | eval=%d | train_groups=%d | eval_groups=%d",
            fold_index,
            n_splits,
            len(train_idx),
            len(test_idx),
            len(np.unique(groups[train_idx])),
            len(np.unique(groups[test_idx])),
        )
        classifier = FlightPhaseClassifier(model_type=model_type, random_state=random_state, **model_kwargs)
        classifier.fit(X[train_idx], y[train_idx])
        predictions = classifier.predict(X[test_idx])
        metadata = [{"source_file": str(group)} for group in groups[test_idx]]
        predictions = smooth_predictions_by_group(predictions, metadata, window_size=smoothing_window)
        metrics = classifier.score_predictions(y[test_idx], predictions)
        fold_rows.append(
            {
                "fold": fold_index,
                "n_train_samples": int(len(train_idx)),
                "n_eval_samples": int(len(test_idx)),
                "n_train_groups": int(len(np.unique(groups[train_idx]))),
                "n_eval_groups": int(len(np.unique(groups[test_idx]))),
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_f1": metrics["macro_f1"],
                "per_class_f1": metrics["per_class_f1"],
                "train_time": metrics["train_time"],
                "inference_time": metrics["inference_time"],
            }
        )

    return {
        "n_splits": int(n_splits),
        "folds": fold_rows,
        "mean_accuracy": float(np.mean([row["accuracy"] for row in fold_rows])),
        "std_accuracy": float(np.std([row["accuracy"] for row in fold_rows])),
        "mean_weighted_f1": float(np.mean([row["weighted_f1"] for row in fold_rows])),
        "std_weighted_f1": float(np.std([row["weighted_f1"] for row in fold_rows])),
        "mean_macro_f1": float(np.mean([row["macro_f1"] for row in fold_rows])),
        "std_macro_f1": float(np.std([row["macro_f1"] for row in fold_rows])),
    }


def compute_learning_curves(
    model_type: str,
    model_kwargs: dict[str, Any],
    random_state: int | None,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    fractions: Iterable[float],
    progress_every: int = 1,
) -> dict[str, Any]:
    unique_classes = max(1, len(np.unique(y_train)))
    curve_rows = []
    valid_fractions = [float(fraction) for fraction in fractions if 0 < float(fraction) <= 1]
    total_fractions = len(valid_fractions)
    progress_every = max(1, int(progress_every))
    loop_start = perf_counter()

    for idx, fraction in enumerate(valid_fractions, start=1):
        n_samples = max(unique_classes, int(round(len(X_train) * fraction)))
        X_subset, y_subset = _sample_training_subset(X_train, y_train, n_samples, random_state)

        if idx == 1 or idx == total_fractions or idx % progress_every == 0:
            elapsed = perf_counter() - loop_start
            LOGGER.info(
                "Learning curve %d/%d | fraction=%.3f | n_samples=%d | elapsed=%.1fs",
                idx,
                total_fractions,
                fraction,
                len(X_subset),
                elapsed,
            )

        classifier = FlightPhaseClassifier(model_type=model_type, random_state=random_state, **model_kwargs)
        classifier.fit(X_subset, y_subset)
        train_predictions = classifier.predict(X_subset)
        subset_train_metrics = classifier.score_predictions(y_subset, train_predictions)
        eval_predictions = classifier.predict(X_eval)
        eval_metrics = classifier.score_predictions(y_eval, eval_predictions)
        curve_rows.append(
            {
                "fraction": fraction,
                "n_samples": int(len(X_subset)),
                "train_accuracy": subset_train_metrics["accuracy"],
                "eval_accuracy": eval_metrics["accuracy"],
                "train_weighted_f1": subset_train_metrics["weighted_f1"],
                "eval_weighted_f1": eval_metrics["weighted_f1"],
            }
        )
    return {"fractions": curve_rows}


def save_training_plots(
    experiment: Experiment,
    classifier: FlightPhaseClassifier,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    curve_results: dict[str, Any],
) -> None:
    experiment.plot_label_distribution(y_train, PHASE_MAPPING, "plots/train_label_distribution.png", "Training label distribution")
    experiment.plot_label_distribution(y_test, PHASE_MAPPING, "plots/test_label_distribution.png", "Test label distribution")
    experiment.plot_confusion_matrix(y_test, y_test_pred, PHASE_LABELS, PHASE_NAMES, "plots/test_confusion_matrix.png", "Test confusion matrix")

    curve_rows = curve_results.get("fractions", [])
    if curve_rows:
        sample_counts = [row["n_samples"] for row in curve_rows]
        experiment.plot_learning_curve_overview(
            sample_counts,
            [row["train_accuracy"] for row in curve_rows],
            [row["eval_accuracy"] for row in curve_rows],
            [row["train_weighted_f1"] for row in curve_rows],
            [row["eval_weighted_f1"] for row in curve_rows],
            "plots/learning_curve.png",
            title="Learning curve overview",
            xlabel="Training samples",
        )
        x_values = [row["fraction"] for row in curve_rows]
        experiment.plot_metric_curve(
            x_values,
            [row["train_accuracy"] for row in curve_rows],
            [row["eval_accuracy"] for row in curve_rows],
            "plots/learning_curve_accuracy.png",
            "Learning curve: accuracy",
            "Accuracy",
        )
        experiment.plot_metric_curve(
            x_values,
            [row["train_weighted_f1"] for row in curve_rows],
            [row["eval_weighted_f1"] for row in curve_rows],
            "plots/learning_curve_weighted_f1.png",
            "Learning curve: weighted F1",
            "Weighted F1",
        )

    loss_curve = _extract_loss_curve(classifier)
    if loss_curve:
        experiment.plot_metric_curve(
            range(1, len(loss_curve) + 1),
            loss_curve,
            loss_curve,
            "plots/loss_curve.png",
            "Loss curve",
            "Loss",
            xlabel="Iteration",
        )
    else:
        experiment.save_text(
            "plots/loss_curve_note.txt",
            "This model does not expose a  loss history. learning curves were saved instead.",
        )


def run_train(args: argparse.Namespace) -> int:
    model_kwargs = parse_model_kwargs(args.model_kwargs)
    dataloader = Dataloader(
        data_path=args.input_dir,
        sequence_length=args.sequence_length,
        stride=args.stride,
        test_size=args.test_size,
        random_state=args.random_state,
        progress_every_files=args.progress_every_files,
        label_strategy=args.label_strategy,
        min_label_purity=args.min_label_purity,
        split_strategy=args.split_strategy,
    )

    experiment = Experiment(args.output_dir, action="train", model_name=args.model)
    config = {
        "command": "train",
        "input_dir": Path(args.input_dir).expanduser().resolve(),
        "model": args.model,
        "model_kwargs": model_kwargs,
        "sequence_length": args.sequence_length,
        "stride": args.stride,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "max_files": args.max_files,
        "curve_fractions": args.curve_fractions,
        "label_strategy": args.label_strategy,
        "min_label_purity": args.min_label_purity,
        "split_strategy": args.split_strategy,
        "smoothing_window": args.smoothing_window,
        "cv_folds": args.cv_folds,
        "stratify": not args.no_stratify,
    }
    experiment.save_json("config.json", config)
    experiment.save_json("memory_estimate.json", dataloader.estimate_memory(n_files=args.max_files))

    try:
        LOGGER.info("Starting data load and train/test split...")
        X_train, X_test, y_train, y_test, train_groups, test_groups, train_metadata, test_metadata = dataloader.get_train_test_split(
            max_files=args.max_files,
            stratify=not args.no_stratify,
            include_groups=True,
            include_metadata=True,
        )
    except ValueError as exc:
        if not args.no_stratify:
            LOGGER.warning("Falling back to a non-stratified split because stratified splitting failed: %s", exc)
            X_train, X_test, y_train, y_test, train_groups, test_groups, train_metadata, test_metadata = dataloader.get_train_test_split(
                max_files=args.max_files,
                stratify=False,
                include_groups=True,
                include_metadata=True,
            )
        else:
            raise

    cv_results = None
    if args.cv_folds and args.cv_folds > 1:
        LOGGER.info("Starting grouped cross-validation with %d folds...", args.cv_folds)
        X_all, y_all, groups_all = dataloader.load_data(
            max_files=args.max_files,
            include_groups=True,
        )
        cv_results = compute_cross_validation(
            model_type=args.model,
            model_kwargs=model_kwargs,
            random_state=args.random_state,
            X=X_all,
            y=y_all,
            groups=groups_all,
            dataloader=dataloader,
            n_splits=args.cv_folds,
            smoothing_window=args.smoothing_window,
        )

    classifier = FlightPhaseClassifier(model_type=args.model, random_state=args.random_state, **model_kwargs)
    try:
        LOGGER.info("Starting model fit for '%s'...", args.model)
        fit_start = perf_counter()
        classifier.fit(X_train, y_train)
        LOGGER.info("Model fit complete in %.1f seconds.", perf_counter() - fit_start)
    except NotImplementedError as exc:
        experiment.save_text("not_implemented.txt", str(exc))
        raise SystemExit(f"Model '{args.model}' is not implemented yet: {exc}") from exc

    model_path = experiment.save_pickle("model.pkl", classifier)
    experiment.save_text(
        "model_saved.txt",
        f"Model was fitted and saved to {model_path.name} before post-training reporting steps ran.\n",
    )

    try:
        LOGGER.info("Running train/test evaluation...")
        y_train_pred_raw = classifier.predict(X_train)
        y_train_pred = smooth_predictions_by_group(y_train_pred_raw, train_metadata, window_size=args.smoothing_window)
        train_metrics = classifier.score_predictions(y_train, y_train_pred)

        y_test_pred_raw = classifier.predict(X_test)
        y_test_pred = smooth_predictions_by_group(y_test_pred_raw, test_metadata, window_size=args.smoothing_window)
        test_metrics = classifier.score_predictions(y_test, y_test_pred)

        metrics_payload = {"train": train_metrics, "test": test_metrics}
        if cv_results is not None:
            metrics_payload["cross_validation"] = {
                "mean_weighted_f1": cv_results["mean_weighted_f1"],
                "std_weighted_f1": cv_results["std_weighted_f1"],
                "mean_macro_f1": cv_results["mean_macro_f1"],
                "std_macro_f1": cv_results["std_macro_f1"],
                "mean_accuracy": cv_results["mean_accuracy"],
                "std_accuracy": cv_results["std_accuracy"],
            }
        metrics_path = experiment.save_json("metrics.json", metrics_payload)
        best_model_result = promote_best_model(
            output_root=args.output_dir,
            run_dir=experiment.run_dir,
            run_model_path=model_path,
            score=test_metrics["weighted_f1"],
            metrics_path=metrics_path,
            config=config,
        )
        experiment.save_json("best_model_status.json", best_model_result)

        LOGGER.info("Computing learning curves for fractions: %s", args.curve_fractions)
        curve_results = compute_learning_curves(
            model_type=args.model,
            model_kwargs=model_kwargs,
            random_state=args.random_state,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_test,
            y_eval=y_test,
            fractions=args.curve_fractions,
            progress_every=args.progress_every_fractions,
        )

        prediction_df = pd.DataFrame(
            {
                "sample_index": np.arange(len(y_test_pred), dtype=int),
                "source_file": [item["source_file"] for item in test_metadata],
                "start_idx": [item["start_idx"] for item in test_metadata],
                "end_idx": [item["end_idx"] for item in test_metadata],
                "label_purity": [item["label_purity"] for item in test_metadata],
                "actual_phase": y_test.astype(int),
                "actual_phase_name": [phase_name(label) for label in y_test],
                "predicted_phase": y_test_pred.astype(int),
                "predicted_phase_name": [phase_name(label) for label in y_test_pred],
                "raw_predicted_phase": y_test_pred_raw.astype(int),
                "raw_predicted_phase_name": [phase_name(label) for label in y_test_pred_raw],
                "is_correct": y_test_pred.astype(int) == y_test.astype(int),
            }
        )
        experiment.save_dataframe("test_predictions.csv", prediction_df)
        experiment.save_json("learning_curves.json", curve_results)
        if cv_results is not None:
            experiment.save_json("cross_validation.json", cv_results)
        experiment.save_json(
            "dataset_summary.json",
            {
                "X_train_shape": X_train.shape,
                "X_test_shape": X_test.shape,
                "y_train_shape": y_train.shape,
                "y_test_shape": y_test.shape,
                "train_group_count": int(len(np.unique(train_groups))),
                "test_group_count": int(len(np.unique(test_groups))),
            },
        )
        save_training_plots(experiment, classifier, y_train, y_test, y_test_pred, curve_results)

        LOGGER.info("Training artifacts and plots saved to: %s", experiment.run_dir)
        print(f"Training complete. Run artifacts saved to: {experiment.run_dir}")
        print(f"Test accuracy: {test_metrics['accuracy']:.4f} | Weighted F1: {test_metrics['weighted_f1']:.4f}")
        if best_model_result["promoted"]:
            print(f"Promoted this run to best model: {best_model_result['best_model_path']}")
        else:
            print(f"Kept existing best model: {best_model_result['best_model_path']}")
        return 0
    except Exception as exc:
        LOGGER.exception("Post-training reporting failed after the model was already saved.")
        experiment.save_text(
            "post_fit_warning.txt",
            "The model finished fitting and was saved to model.pkl, but a later reporting step failed.\n\n"
            + traceback.format_exc(),
        )
        print(f"Training finished and the model was saved to: {experiment.run_dir}")
        print(f"Warning: post-training reporting failed after the model was saved: {exc}")
        return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, evaluate, and compare rocket flight-phase classifiers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model from a folder of CSV files.")
    train_parser.add_argument("--input-dir", required=True, help="Folder containing training CSV files.")
    train_parser.add_argument("--output-dir", default="outputs", help="Root directory where run artifacts are written.")
    train_parser.add_argument("--model", required=True, choices=sorted(FlightPhaseClassifier.SUPPORTED_MODEL_TYPES),
                              help="Model type to train.")
    train_parser.add_argument("--model-kwargs", nargs="*", default=[], metavar="KEY=VALUE",
                              help="Optional model-specific hyperparameters, e.g. max_iter=2000 C=0.5")
    train_parser.add_argument("--sequence-length", type=int, default=100, help="Sliding window length in timesteps.")
    train_parser.add_argument("--stride", type=int, default=50, help="Sliding window stride in timesteps.")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of windows reserved for testing.")
    train_parser.add_argument("--max-files", type=int, default=None,
                              help="Optional cap on the number of CSV files to load.")
    train_parser.add_argument("--random-state", type=int, default=42,
                              help="Random seed for splitting and any stochastic model behavior.")
    train_parser.add_argument("--curve-fractions", type=float, nargs="+", default=DEFAULT_CURVE_FRACTIONS,
                              help="Fractions of the training set used to generate learning curves.")
    train_parser.add_argument("--progress-every-files", type=int, default=100,
                              help="Print/load progress every N CSV files while creating windows.")
    train_parser.add_argument("--progress-every-fractions", type=int, default=1,
                              help="Print progress every N learning-curve fractions.")
    train_parser.add_argument("--label-strategy", choices=sorted(Dataloader.SUPPORTED_LABEL_STRATEGIES), default="center",
                              help="How each sliding window is labeled. 'center' usually works better for short transitions.")
    train_parser.add_argument("--min-label-purity", type=float, default=0.0,
                              help="Drop training windows whose selected label covers less than this fraction of the window.")
    train_parser.add_argument("--split-strategy", choices=["group", "window"], default="group",
                              help="How to split train/test data. 'group' keeps entire flights in one split.")
    train_parser.add_argument("--smoothing-window", type=int, default=5,
                              help="Odd-sized majority filter applied per flight to reduce flickering predictions.")
    train_parser.add_argument("--cv-folds", type=int, default=0,
                              help="Optional grouped cross-validation fold count. Disabled when set to 0 or 1.")
    train_parser.add_argument("--no-stratify", action="store_true", help="Disable stratified train/test splitting.")
    train_parser.set_defaults(func=run_train)

    classify_parser = subparsers.add_parser("classify", help="Load a saved model and classify one CSV file.")
    classify_parser.add_argument("--model-path", required=True, help="Path to a saved model.pkl or its run directory.")
    classify_parser.add_argument("--input-file", required=True, help="CSV file to classify.")
    classify_parser.add_argument("--output-dir", default="outputs",
                                 help="Root directory where run artifacts are written.")
    classify_parser.add_argument("--sequence-length", type=int, default=None,
                                 help="Override the saved sequence length used for sliding windows.")
    classify_parser.add_argument("--stride", type=int, default=None,
                                 help="Override the saved stride used for sliding windows.")
    classify_parser.add_argument("--random-state", type=int, default=42,
                                 help="Random seed used for loader construction.")
    classify_parser.add_argument("--label-strategy", choices=sorted(Dataloader.SUPPORTED_LABEL_STRATEGIES), default=None,
                                 help="Optional override for how windows are labeled before evaluation.")
    classify_parser.add_argument("--min-label-purity", type=float, default=0.0,
                                 help="Optional override for dropping ambiguous windows before evaluation.")
    classify_parser.add_argument("--smoothing-window", type=int, default=None,
                                 help="Optional override for per-flight prediction smoothing.")
    classify_parser.set_defaults(func=run_classify)

    compare_parser = subparsers.add_parser("compare", help="Compare two saved models on a labeled CSV file.")
    compare_parser.add_argument("--model-a", required=True, help="Path to the first saved model.pkl or run directory.")
    compare_parser.add_argument("--model-b", required=True, help="Path to the second saved model.pkl or run directory.")
    compare_parser.add_argument("--input-file", required=True, help="CSV file to classify with both models.")
    compare_parser.add_argument("--output-dir", default="outputs",
                                help="Root directory where run artifacts are written.")
    compare_parser.add_argument("--sequence-length", type=int, default=None,
                                help="Override the saved sequence length used for sliding windows.")
    compare_parser.add_argument("--stride", type=int, default=None,
                                help="Override the saved stride used for sliding windows.")
    compare_parser.add_argument("--random-state", type=int, default=42,
                                help="Random seed used for loader construction.")
    compare_parser.set_defaults(func=run_compare)


    args = parser.parse_args()
    args.func(args)

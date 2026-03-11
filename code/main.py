import argparse
import logging
import traceback

from pathlib import Path
from typing import Iterable

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

    dataloader = Dataloader(
        data_path=str(input_file.parent),
        sequence_length=sequence_length,
        stride=stride,
        random_state=args.random_state,
    )
    X, y, metadata = dataloader.load_file(input_file, include_metadata=True)

    predictions = classifier.predict(X)
    evaluation = classifier.evaluate(X, y)

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
            "random_state": args.random_state,
        },
    )

    prediction_df = build_predictions_dataframe(metadata, predictions, y)
    experiment.save_dataframe("predictions.csv", prediction_df)
    experiment.save_json("metrics.json", evaluation)
    experiment.plot_confusion_matrix(y, predictions, PHASE_LABELS, PHASE_NAMES, "plots/confusion_matrix.png",
                                     f"Confusion matrix: {input_file.name}")
    x_axis, xlabel = select_x_axis(metadata)
    experiment.plot_phase_timeline(
        x_axis,
        predictions,
        PHASE_MAPPING,
        "plots/prediction_timeline.png",
        f"Prediction timeline: {input_file.name}",
        ground_truth=y,
        xlabel=xlabel,
    )
    experiment.plot_label_distribution(predictions, PHASE_MAPPING, "plots/predicted_label_distribution.png",
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


def compute_learning_curves(
    model_type: str,
    model_kwargs: dict[str, Any],
    random_state: int | None,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    fractions: Iterable[float],
) -> dict[str, Any]:
    unique_classes = max(1, len(np.unique(y_train)))
    curve_rows = []
    for fraction in fractions:
        fraction = float(fraction)
        if fraction <= 0 or fraction > 1:
            continue
        n_samples = max(unique_classes, int(round(len(X_train) * fraction)))
        X_subset, y_subset = _sample_training_subset(X_train, y_train, n_samples, random_state)

        classifier = FlightPhaseClassifier(model_type=model_type, random_state=random_state, **model_kwargs)
        classifier.fit(X_subset, y_subset)
        subset_train_metrics = classifier.evaluate(X_subset, y_subset)
        eval_metrics = classifier.evaluate(X_eval, y_eval)
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
        "stratify": not args.no_stratify,
    }
    experiment.save_json("config.json", config)
    experiment.save_json("memory_estimate.json", dataloader.estimate_memory(n_files=args.max_files))

    try:
        X_train, X_test, y_train, y_test = dataloader.get_train_test_split(
            max_files=args.max_files,
            stratify=not args.no_stratify,
        )
    except ValueError as exc:
        if not args.no_stratify:
            LOGGER.warning("Falling back to a non-stratified split because stratified splitting failed: %s", exc)
            X_train, X_test, y_train, y_test = dataloader.get_train_test_split(
                max_files=args.max_files,
                stratify=False,
            )
        else:
            raise

    classifier = FlightPhaseClassifier(model_type=args.model, random_state=args.random_state, **model_kwargs)
    try:
        classifier.fit(X_train, y_train)
    except NotImplementedError as exc:
        experiment.save_text("not_implemented.txt", str(exc))
        raise SystemExit(f"Model '{args.model}' is not implemented yet: {exc}") from exc

    model_path = experiment.save_pickle("model.pkl", classifier)
    experiment.save_text(
        "model_saved.txt",
        f"Model was fitted and saved to {model_path.name} before post-training reporting steps ran.\n",
    )

    try:
        train_metrics = classifier.evaluate(X_train, y_train)
        test_metrics = classifier.evaluate(X_test, y_test)
        y_test_pred = classifier.predict(X_test)

        metrics_payload = {"train": train_metrics, "test": test_metrics}
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

        curve_results = compute_learning_curves(
            model_type=args.model,
            model_kwargs=model_kwargs,
            random_state=args.random_state,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_test,
            y_eval=y_test,
            fractions=args.curve_fractions,
        )

        prediction_df = pd.DataFrame(
            {
                "sample_index": np.arange(len(y_test_pred), dtype=int),
                "actual_phase": y_test.astype(int),
                "actual_phase_name": [phase_name(label) for label in y_test],
                "predicted_phase": y_test_pred.astype(int),
                "predicted_phase_name": [phase_name(label) for label in y_test_pred],
                "is_correct": y_test_pred.astype(int) == y_test.astype(int),
            }
        )
        experiment.save_dataframe("test_predictions.csv", prediction_df)
        experiment.save_json("learning_curves.json", curve_results)
        experiment.save_json(
            "dataset_summary.json",
            {
                "X_train_shape": X_train.shape,
                "X_test_shape": X_test.shape,
                "y_train_shape": y_train.shape,
                "y_test_shape": y_test.shape,
            },
        )
        save_training_plots(experiment, classifier, y_train, y_test, y_test_pred, curve_results)

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

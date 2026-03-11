import argparse
import PhaseClassifier

PHASE_MAPPING = PhaseClassifier.PHASE_MAPPING
PHASE_LABELS = sorted(PHASE_MAPPING)
PHASE_NAMES = [PHASE_MAPPING[label] for label in PHASE_LABELS]
DEFAULT_CURVE_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

def run_classify(args: argparse.Namespace) -> int:
    pass

def run_compare(args: argparse.Namespace) -> int:
    pass

def run_train(args: argparse.Namespace) -> int:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, evaluate, and compare rocket flight-phase classifiers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model from a folder of CSV files.")
    train_parser.add_argument("--input-dir", required=True, help="Folder containing training CSV files.")
    train_parser.add_argument("--output-dir", default="outputs", help="Root directory where run artifacts are written.")
    train_parser.add_argument("--model", required=True, choices=sorted(PhaseClassifier.SUPPORTED_MODEL_TYPES),
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

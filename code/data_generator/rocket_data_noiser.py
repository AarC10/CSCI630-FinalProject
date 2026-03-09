"""
This will inject Gaussian noise into the sensor cols, drop physisc derived cols and collapse the event labels into a flight phase column
"""

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s - %(message)s",
)

# Default noise params based on typical COTS sensors
# BMP388 ~10 Pa, 0.5K
# LSM6DSL 90e-6 g/sqrt(Hz) accel, 4 mdps/sqrt(Hz) gyro @ 100 Hz
DEFAULT_NOISE_CONFIG = {
    "AIR_PRESSURE": {
        "std": 10.0,
        "bias": 0.0,
    },
    "AIR_TEMPERATURE": {
        "std": 0.5,
        "bias": 0.0,
    },
    "ACCELERATION_XY": {
        "std": 0.00883,
        "bias": 0.0,
    },
    "ACCELERATION_Z": {
        "std": 0.00883,
        "bias": 0.0,
    },
    "ROLL_RATE": {
        "std": 0.000698,
        "bias": 0.0,
    },
    "PITCH_RATE": {
        "std": 0.000698,
        "bias": 0.0,
    },
    "YAW_RATE": {
        "std": 0.000698,
        "bias": 0.0,
    },
}

# Columns that must NEVER be modified (but are kept in output)
PROTECTED_COLS = {
    "TIME",
}

# Columns to drop from output — physics-derived, not real sensor outputs
DROP_COLS = {
    "ALTITUDE",
    "VELOCITY_XY",
    "VELOCITY_Z",
}

# Priority-ordered list of events — first match wins when multiple are active.
# 0 = no_event (fallback)
EVENT_PRIORITY = [
    "event_liftoff",
    "event_burnout",
    "event_apogee",
    "event_recovery_deployment",
    "event_landing",
]

# Integer label for each event. 0 is reserved for no_event.
EVENT_LABEL_MAP = {name: i + 1 for i, name in enumerate(EVENT_PRIORITY)}
EVENT_LABEL_MAP["no_event"] = 0
# 0: no_event, 1: liftoff, 2: burnout, 3: apogee, 4: recovery_deployment, 5: landing


def inject_noise_to_file(
    input_path: Path,
    output_dir: Path,
    noise_config: dict,
    seed: int | None,
) -> str:
    """
    Loads a single CSV, injects noise, write to output_dir.
    Returns the output filename on success, raises on failure.
    """
    df = pd.read_csv(input_path)

    # Derive a per-file seed from the global seed + filename hash
    if seed is not None:
        file_seed = (seed + hash(input_path.name)) & 0xFFFFFFFF
        rng = np.random.default_rng(file_seed)
    else:
        rng = np.random.default_rng()

    for col, params in noise_config.items():
        if col not in df.columns:
            continue
        if col in PROTECTED_COLS:
            logging.warning(f"Skipping protected column '{col}' in {input_path.name}")
            continue
        if col.startswith("event_"):
            logging.warning(f"Skipping event label column '{col}' in {input_path.name}")
            continue

        std = float(params.get("std", 0.0))
        bias = float(params.get("bias", 0.0))

        noise = rng.normal(loc=bias, scale=std, size=len(df))
        df[col] = df[col] + noise

    # Collapse multi-label binary event columns into a single integer label.
    # If apogee and reco deployment share the same window, active rows can be identical
    # Therefore, if both active, split the overlap so apogee gets the first half
    present_events = [e for e in EVENT_PRIORITY if e in df.columns]

    if "event_apogee" in df.columns and "event_recovery_deployment" in df.columns:
        overlap_mask = (df["event_apogee"] == 1) & (df["event_recovery_deployment"] == 1)
        overlap_indices = df.index[overlap_mask].tolist()

        if overlap_indices:
            split = len(overlap_indices) // 2
            apogee_only = overlap_indices[:split]
            recovery_only = overlap_indices[split:]
            df.loc[apogee_only, "event_recovery_deployment"] = 0
            df.loc[recovery_only, "event_apogee"] = 0

    def resolve_label(row):
        for event_col in present_events:
            if row[event_col] == 1:
                return EVENT_LABEL_MAP[event_col]
        return 0

    if present_events:
        df["flight_phase"] = df[present_events].apply(resolve_label, axis=1)
        df = df.drop(columns=present_events)

    leftover_event_cols = [c for c in df.columns if c.startswith("event_")]
    if leftover_event_cols:
        df = df.drop(columns=leftover_event_cols)

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    out_path = output_dir / input_path.name
    df.to_csv(out_path, index=False)
    return str(out_path)


def _worker(args):
    input_path, output_dir, noise_config, seed = args
    return inject_noise_to_file(input_path, output_dir, noise_config, seed)


def run(input_dir: str, output_dir: str, noise_config: dict, workers: int, seed: int | None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_path.glob("*.csv"))
    if not csv_files:
        logging.error(f"No CSV files found in {input_dir}")
        return

    logging.info(f"Found {len(csv_files)} CSV files. Workers: {workers}. Seed: {seed}")
    logging.info(f"Noise config: {json.dumps(noise_config, indent=2)}")

    tasks = [(f, output_path, noise_config, seed) for f in csv_files]

    success, failure = 0, 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, t): t[0] for t in tasks}
        for future in as_completed(futures):
            src = futures[future]
            try:
                out = future.result()
                success += 1
                if success % 100 == 0:
                    logging.info(f"Progress: {success}/{len(csv_files)} done")
            except Exception as e:
                failure += 1
                logging.error(f"Failed on {src.name}: {e}")

    logging.info(f"Done. {success} succeeded, {failure} failed. Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Inject Gaussian sensor noise into rocket flight CSV files.")
    parser.add_argument("--input_dir",  required=True,  help="Folder containing input CSVs")
    parser.add_argument("--output_dir", required=True,  help="Folder to write noisy CSVs")
    parser.add_argument("--workers",    type=int, default=os.cpu_count(),
                        help="Number of parallel workers")
    parser.add_argument("--seed",       type=int, default=None,
                        help="Global random seed for reproducibility")
    parser.add_argument("--noise_config", type=str, default=None,
                        help="Path to JSON file overriding default noise parameters")
    args = parser.parse_args()

    noise_config = DEFAULT_NOISE_CONFIG.copy()
    if args.noise_config:
        with open(args.noise_config) as f:
            overrides = json.load(f)
        noise_config.update(overrides)
        logging.info(f"Loaded noise config overrides from {args.noise_config}")

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        noise_config=noise_config,
        workers=args.workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
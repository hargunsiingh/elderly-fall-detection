from __future__ import annotations

import argparse

import pandas as pd

from .imu_features import make_feature_table
from .imu_model import save_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IMU-based elderly fall detection model")
    parser.add_argument("--input", required=True, help="CSV with ax, ay, az, gx, gy, gz and optional label")
    parser.add_argument("--output", default="models/imu_fall_model.joblib", help="Output model path")
    parser.add_argument("--window-size", type=int, default=50, help="Samples per feature window")
    parser.add_argument("--step", type=int, default=10, help="Sliding window step")
    parser.add_argument("--label-column", default="label", help="Label column; use blank for anomaly-only data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_column = args.label_column or None
    frame = pd.read_csv(args.input)
    features = make_feature_table(
        frame,
        window_size=args.window_size,
        step=args.step,
        label_column=label_column,
    )
    model = train_model(features)
    save_model(model, args.output)
    print(f"Saved {model.mode} model to {args.output}")
    print(f"Feature windows: {len(features)}")


if __name__ == "__main__":
    main()

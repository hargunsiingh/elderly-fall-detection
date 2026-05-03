from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

IMU_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]


def validate_imu_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in IMU_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required IMU columns: {', '.join(missing)}")


def extract_features(window: pd.DataFrame) -> dict[str, float]:
    validate_imu_frame(window)
    data = window[IMU_COLUMNS].astype(float)

    acc = data[["ax", "ay", "az"]].to_numpy()
    gyro = data[["gx", "gy", "gz"]].to_numpy()
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    jerk = np.diff(acc_mag, prepend=acc_mag[0])

    features: dict[str, float] = {
        "acc_mag_mean": float(np.mean(acc_mag)),
        "acc_mag_std": float(np.std(acc_mag)),
        "acc_mag_min": float(np.min(acc_mag)),
        "acc_mag_max": float(np.max(acc_mag)),
        "acc_mag_range": float(np.ptp(acc_mag)),
        "gyro_mag_mean": float(np.mean(gyro_mag)),
        "gyro_mag_std": float(np.std(gyro_mag)),
        "gyro_mag_max": float(np.max(gyro_mag)),
        "jerk_mean_abs": float(np.mean(np.abs(jerk))),
        "jerk_max_abs": float(np.max(np.abs(jerk))),
        "posture_ax_change": float(abs(data["ax"].iloc[-1] - data["ax"].iloc[0])),
        "posture_ay_change": float(abs(data["ay"].iloc[-1] - data["ay"].iloc[0])),
        "posture_az_change": float(abs(data["az"].iloc[-1] - data["az"].iloc[0])),
    }

    for column in IMU_COLUMNS:
        values = data[column].to_numpy()
        features[f"{column}_mean"] = float(np.mean(values))
        features[f"{column}_std"] = float(np.std(values))
        features[f"{column}_max_abs"] = float(np.max(np.abs(values)))

    return features


def sliding_windows(frame: pd.DataFrame, window_size: int, step: int) -> Iterable[pd.DataFrame]:
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive")

    for start in range(0, max(len(frame) - window_size + 1, 0), step):
        yield frame.iloc[start : start + window_size]


def make_feature_table(
    frame: pd.DataFrame,
    *,
    window_size: int = 50,
    step: int = 10,
    label_column: str | None = "label",
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    has_label = label_column is not None and label_column in frame.columns

    for window in sliding_windows(frame, window_size, step):
        row = extract_features(window)
        if has_label:
            row["label"] = int(_window_has_fall(window[label_column]))
        rows.append(row)

    if not rows:
        raise ValueError("Not enough IMU rows to create one feature window")

    return pd.DataFrame(rows)


def _window_has_fall(labels: pd.Series) -> bool:
    normalized = labels.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "fall", "fallen", "anomaly"}).any()

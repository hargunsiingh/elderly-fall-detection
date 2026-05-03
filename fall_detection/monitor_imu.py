from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from time import sleep

import pandas as pd

from .alerts import AlertLogger
from .imu_features import IMU_COLUMNS, extract_features
from .imu_model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Raspberry Pi IMU stream for elderly falls")
    parser.add_argument("--model", default="models/imu_fall_model.joblib", help="Trained model path")
    parser.add_argument("--serial-port", help="Serial port from ESP32/Arduino, for example COM3")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--csv", help="Replay a CSV file instead of using serial hardware")
    parser.add_argument("--window-size", type=int, default=50, help="Samples per inference window")
    parser.add_argument("--step", type=int, default=10, help="Run inference every N samples")
    parser.add_argument("--alert-log", default="alerts/imu_fall_alerts.csv", help="CSV alert log path")
    parser.add_argument("--delay", type=float, default=0.02, help="Replay delay for CSV rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.serial_port and not args.csv:
        raise RuntimeError("Provide either --serial-port or --csv")

    model = load_model(args.model)
    logger = AlertLogger(args.alert_log)
    buffer: deque[dict[str, float]] = deque(maxlen=args.window_size)
    sample_count = 0

    for sample in _read_samples(args):
        buffer.append(sample)
        sample_count += 1
        if len(buffer) < args.window_size or sample_count % args.step != 0:
            continue

        features = extract_features(pd.DataFrame(buffer))
        prediction = model.predict(features)
        status = "FALL" if prediction.is_fall else "normal"
        print(f"{datetime.now().isoformat(timespec='seconds')} | {status} | confidence={prediction.confidence}")

        if prediction.is_fall:
            logger.log(datetime.now().isoformat(timespec="seconds"), _to_event(prediction))


def _read_samples(args: argparse.Namespace):
    if args.csv:
        frame = pd.read_csv(args.csv)
        for _, row in frame.iterrows():
            yield {column: float(row[column]) for column in IMU_COLUMNS}
            sleep(args.delay)
        return

    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is required for --serial-port. Install dependencies first.") from exc

    with serial.Serial(args.serial_port, args.baud, timeout=1) as port:
        while True:
            line = port.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            values = [value.strip() for value in line.split(",")]
            if len(values) != len(IMU_COLUMNS):
                continue
            yield {column: float(value) for column, value in zip(IMU_COLUMNS, values)}


def _to_event(prediction):
    from .detector import FallEvent

    return FallEvent(
        is_fall=prediction.is_fall,
        confidence=prediction.confidence,
        reason=f"IMU {prediction.mode} model",
        torso_angle=0.0,
        aspect_ratio=0.0,
        vertical_velocity=0.0,
    )


if __name__ == "__main__":
    main()

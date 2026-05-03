from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate demo MPU6050-style IMU data")
    parser.add_argument("--output", default="data/sample_imu.csv", help="Output CSV path")
    parser.add_argument("--samples", type=int, default=1200, help="Number of samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = generate_sample_imu(args.samples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Saved demo IMU data to {output}")


def generate_sample_imu(samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(samples)

    ax = rng.normal(0.02, 0.04, samples)
    ay = rng.normal(0.01, 0.04, samples)
    az = rng.normal(1.0, 0.05, samples)
    gx = rng.normal(0.0, 3.0, samples)
    gy = rng.normal(0.0, 3.0, samples)
    gz = rng.normal(0.0, 3.0, samples)
    label = np.zeros(samples, dtype=int)

    for center in (360, 830):
        span = slice(center, min(center + 45, samples))
        impact = np.exp(-((t[span] - center - 8) ** 2) / 18)
        ax[span] += np.linspace(0.1, 1.1, len(t[span]))
        ay[span] += rng.normal(0.5, 0.18, len(t[span]))
        az[span] -= np.linspace(0.0, 0.9, len(t[span]))
        gx[span] += 90 * impact + rng.normal(0, 8, len(t[span]))
        gy[span] += 120 * impact + rng.normal(0, 8, len(t[span]))
        gz[span] += 60 * impact + rng.normal(0, 8, len(t[span]))
        label[span] = 1

    return pd.DataFrame(
        {
            "timestamp_ms": t * 20,
            "ax": ax,
            "ay": ay,
            "az": az,
            "gx": gx,
            "gy": gy,
            "gz": gz,
            "label": label,
        }
    )


if __name__ == "__main__":
    main()

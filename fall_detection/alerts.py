from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from .detector import FallEvent


class AlertLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def log(self, timestamp: str, event: FallEvent) -> None:
        row = {"timestamp": timestamp, **asdict(event)}
        with self.path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            writer.writerow(row)

    def _ensure_header(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return

        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "timestamp",
                    "is_fall",
                    "confidence",
                    "reason",
                    "torso_angle",
                    "aspect_ratio",
                    "vertical_velocity",
                ]
            )

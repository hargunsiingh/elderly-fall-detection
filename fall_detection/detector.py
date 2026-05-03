from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import monotonic

from .geometry import Point, angle_from_vertical, bounding_box, midpoint


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    visibility: float = 1.0

    @property
    def point(self) -> Point:
        return Point(self.x, self.y)


@dataclass(frozen=True)
class BodyObservation:
    left_shoulder: Keypoint
    right_shoulder: Keypoint
    left_hip: Keypoint
    right_hip: Keypoint
    left_ankle: Keypoint
    right_ankle: Keypoint
    timestamp: float | None = None


@dataclass(frozen=True)
class FallEvent:
    is_fall: bool
    confidence: float
    reason: str
    torso_angle: float
    aspect_ratio: float
    vertical_velocity: float


@dataclass(frozen=True)
class _FrameMetrics:
    timestamp: float
    torso_angle: float
    aspect_ratio: float
    hip_y: float
    centroid_y: float


class FallDetector:
    """Stateful pose-geometry fall detector.

    Coordinates are normalized to the image, with y increasing downward.
    """

    def __init__(
        self,
        *,
        horizontal_angle_threshold: float = 58.0,
        aspect_ratio_threshold: float = 1.12,
        hip_low_threshold: float = 0.62,
        drop_velocity_threshold: float = 0.75,
        min_confidence: float = 0.55,
        cooldown_seconds: float = 2.5,
        history_size: int = 12,
    ) -> None:
        self.horizontal_angle_threshold = horizontal_angle_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.hip_low_threshold = hip_low_threshold
        self.drop_velocity_threshold = drop_velocity_threshold
        self.min_confidence = min_confidence
        self.cooldown_seconds = cooldown_seconds
        self.history: deque[_FrameMetrics] = deque(maxlen=history_size)
        self._last_alert_at = -cooldown_seconds

    def update(self, observation: BodyObservation) -> FallEvent:
        metrics = self._to_metrics(observation)
        velocity = self._vertical_velocity(metrics)
        self.history.append(metrics)

        flat_posture = metrics.torso_angle >= self.horizontal_angle_threshold
        wide_body = metrics.aspect_ratio >= self.aspect_ratio_threshold
        low_hips = metrics.hip_y >= self.hip_low_threshold
        fast_drop = velocity >= self.drop_velocity_threshold

        confidence = 0.0
        confidence += 0.38 if flat_posture else 0.0
        confidence += 0.24 if wide_body else 0.0
        confidence += 0.20 if low_hips else 0.0
        confidence += 0.18 if fast_drop else 0.0

        likely_fall = flat_posture and (wide_body or low_hips or fast_drop)
        outside_cooldown = metrics.timestamp - self._last_alert_at >= self.cooldown_seconds
        is_fall = likely_fall and confidence >= self.min_confidence and outside_cooldown

        if is_fall:
            self._last_alert_at = metrics.timestamp

        reason = self._reason(flat_posture, wide_body, low_hips, fast_drop, outside_cooldown)
        return FallEvent(
            is_fall=is_fall,
            confidence=round(min(confidence, 1.0), 3),
            reason=reason,
            torso_angle=round(metrics.torso_angle, 2),
            aspect_ratio=round(metrics.aspect_ratio, 2),
            vertical_velocity=round(velocity, 2),
        )

    def _to_metrics(self, observation: BodyObservation) -> _FrameMetrics:
        shoulders = midpoint(observation.left_shoulder.point, observation.right_shoulder.point)
        hips = midpoint(observation.left_hip.point, observation.right_hip.point)
        points = [
            observation.left_shoulder.point,
            observation.right_shoulder.point,
            observation.left_hip.point,
            observation.right_hip.point,
            observation.left_ankle.point,
            observation.right_ankle.point,
        ]
        min_x, min_y, max_x, max_y = bounding_box(points)
        width = max(max_x - min_x, 0.001)
        height = max(max_y - min_y, 0.001)
        timestamp = observation.timestamp if observation.timestamp is not None else monotonic()

        return _FrameMetrics(
            timestamp=timestamp,
            torso_angle=angle_from_vertical(shoulders, hips),
            aspect_ratio=width / height,
            hip_y=hips.y,
            centroid_y=(min_y + max_y) / 2.0,
        )

    def _vertical_velocity(self, current: _FrameMetrics) -> float:
        if not self.history:
            return 0.0

        previous = self.history[0]
        elapsed = max(current.timestamp - previous.timestamp, 0.001)
        return (current.centroid_y - previous.centroid_y) / elapsed

    @staticmethod
    def _reason(
        flat_posture: bool,
        wide_body: bool,
        low_hips: bool,
        fast_drop: bool,
        outside_cooldown: bool,
    ) -> str:
        signals: list[str] = []
        if flat_posture:
            signals.append("horizontal posture")
        if wide_body:
            signals.append("wide body box")
        if low_hips:
            signals.append("low hip position")
        if fast_drop:
            signals.append("rapid downward movement")
        if not outside_cooldown:
            signals.append("cooldown active")
        return ", ".join(signals) if signals else "normal posture"

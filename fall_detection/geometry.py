from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def midpoint(a: Point, b: Point) -> Point:
    return Point((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def distance(a: Point, b: Point) -> float:
    return hypot(a.x - b.x, a.y - b.y)


def angle_from_vertical(top: Point, bottom: Point) -> float:
    """Return torso angle in degrees, where 0 is upright and 90 is horizontal."""
    dx = bottom.x - top.x
    dy = bottom.y - top.y
    return abs(degrees(atan2(dx, dy)))


def bounding_box(points: list[Point]) -> tuple[float, float, float, float]:
    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)
    return min_x, min_y, max_x, max_y

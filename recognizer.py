"""Template-based $P recognizer for trajectories of isolated digits."""

from collections.abc import Sequence
from math import dist, isfinite, isqrt

type Point = tuple[float, float]
type Strokes = Sequence[Sequence[Point]]
type Cloud = list[Point]


def resample(strokes: Strokes, n: int = 32) -> Cloud:
    """Sample at equal distances along the drawn segments."""
    if n < 2:
        raise ValueError("Resampling requires at least two points")

    segments = []
    total = 0.0
    for stroke in strokes:
        for start, end in zip(stroke, stroke[1:]):
            length = dist(start, end)
            if length > 0:
                segments.append((start, end, total, total + length))
                total += length
    if not segments:
        raise ValueError("Нарисуйте цифру")
    if not isfinite(total):
        raise ValueError("Некорректная длина штрихов")

    points = [segments[0][0]]
    segment_index = 0
    for index in range(1, n - 1):
        target = total * (index / (n - 1))
        while segments[segment_index][3] < target:
            segment_index += 1
        start, end, before, after = segments[segment_index]
        fraction = (target - before) / (after - before)
        points.append((
            start[0] + fraction * (end[0] - start[0]),
            start[1] + fraction * (end[1] - start[1]),
        ))
    points.append(segments[-1][1])
    return points


def normalize(strokes: Strokes) -> Cloud:
    points = resample(strokes)
    xs, ys = zip(*points)
    min_x, min_y = min(xs), min(ys)
    scale = max(max(xs) - min_x, max(ys) - min_y)
    if not isfinite(scale) or scale == 0:
        raise ValueError("Некорректный размер рисунка")
    scaled = [((x - min_x) / scale, (y - min_y) / scale) for x, y in points]
    center_x = sum(x for x, _ in scaled) / len(scaled)
    center_y = sum(y for _, y in scaled) / len(scaled)
    return [(x - center_x, y - center_y) for x, y in scaled]


def cloud_distance(a: Cloud, b: Cloud, start: int) -> float:
    n = len(a)
    used = [False] * n
    total = 0.0
    for offset in range(n):
        point = a[(start + offset) % n]
        distance, index = min(
            (dist(point, other), j)
            for j, other in enumerate(b)
            if not used[j]
        )
        used[index] = True
        total += (1 - offset / n) * distance
    return total


def cloud_match(a: Cloud, b: Cloud) -> float:
    step = isqrt(len(a))
    return min(
        min(cloud_distance(a, b, start), cloud_distance(b, a, start))
        for start in range(0, len(a), step)
    )


class Recognizer:
    def __init__(self) -> None:
        self.templates: list[tuple[str, Cloud]] = []

    def predict(self, strokes: Strokes) -> str:
        cloud = normalize(strokes)
        label, _ = min(self.templates, key=lambda template: cloud_match(cloud, template[1]))
        return label

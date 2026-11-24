"""Built-in digit trajectories and the user's raw labeled samples."""

import json
from math import isfinite
from pathlib import Path
from typing import TypedDict

from PySide6.QtCore import QIODevice, QLockFile, QSaveFile

from recognizer import Point, Strokes, normalize


class Sample(TypedDict):
    label: str
    author: str
    strokes: list[list[Point]]


# Two manually drawn paths per digit in a common coordinate system.
DEMO: dict[str, tuple[Strokes, Strokes]] = {
    "0": (
        [[(5, 1), (2, 2), (1, 5), (1, 8), (3, 10), (6, 10), (8, 8), (8, 4), (7, 2), (5, 1)]],
        [[(6, 1), (3, 1), (1, 4), (1, 8), (3, 10), (6, 9), (8, 6), (8, 3), (6, 1)]],
    ),
    "1": (
        [[(4, 1), (4, 10)]],
        [[(2, 4), (5, 1), (5, 10)], [(2, 10), (8, 10)]],
    ),
    "2": (
        [[(1, 3), (3, 1), (6, 1), (8, 3), (7, 5), (1, 10), (8, 10)]],
        [[(1, 2), (4, 1), (7, 2), (7, 4), (5, 6), (1, 9), (5, 10), (8, 9)]],
    ),
    "3": (
        [[(1, 2), (4, 1), (7, 2), (7, 4), (4, 5), (7, 6), (8, 8), (6, 10), (3, 10), (1, 9)]],
        [[(1, 1), (7, 1), (4, 5), (7, 5), (8, 7), (7, 9), (4, 10), (1, 9)]],
    ),
    "4": (
        [[(3, 0), (0, 4), (4, 4)], [(3, 0), (3, 6)]],
        [[(3, 1), (1, 6), (8, 6)], [(6, 2), (6, 10)]],
    ),
    "5": (
        [[(8, 1), (2, 1), (1, 5), (5, 4), (8, 6), (8, 8), (6, 10), (3, 10), (1, 9)]],
        [[(3, 1), (2, 5), (5, 4), (7, 5), (8, 7), (7, 9), (4, 10), (1, 8)], [(3, 1), (8, 1)]],
    ),
    "6": (
        [[(7, 1), (4, 2), (2, 5), (1, 8), (3, 10), (6, 10), (8, 8), (7, 6), (4, 5), (2, 6)]],
        [[(8, 2), (5, 1), (3, 3), (1, 6), (1, 9), (4, 10), (7, 9), (8, 7), (6, 5), (3, 5), (1, 7)]],
    ),
    "7": (
        [[(1, 1), (8, 1), (3, 10)]],
        [[(1, 2), (8, 1), (4, 10)], [(2, 5), (8, 5)]],
    ),
    "8": (
        [[(5, 5), (2, 3), (3, 1), (6, 1), (8, 3), (6, 5), (2, 7), (1, 9), (4, 10), (7, 10), (8, 8), (5, 5)]],
        [[(5, 5), (2, 4), (2, 2), (5, 1), (7, 2), (7, 4), (5, 5)],
         [(5, 5), (8, 7), (7, 9), (4, 10), (1, 8), (2, 6), (5, 5)]],
    ),
    "9": (
        [[(7, 6), (4, 6), (1, 4), (2, 2), (5, 1), (7, 2), (8, 4), (7, 7), (5, 10)]],
        [[(7, 6), (4, 7), (1, 5), (1, 3), (3, 1), (6, 1), (8, 3), (7, 7), (7, 10)]],
    ),
}


def _parse_sample(data: object) -> Sample:
    if not isinstance(data, dict):
        raise ValueError("Ожидалась запись образца")
    label, author, raw_strokes = data.get("label"), data.get("author"), data.get("strokes")
    if not isinstance(label, str) or label not in DEMO:
        raise ValueError("Некорректная метка цифры")
    if not isinstance(author, str) or not author.strip():
        raise ValueError("У образца отсутствует автор")
    if not isinstance(raw_strokes, list):
        raise ValueError("У образца отсутствуют штрихи")

    strokes = []
    for raw_stroke in raw_strokes:
        if not isinstance(raw_stroke, list) or not raw_stroke:
            raise ValueError("Некорректный штрих")
        stroke = []
        for point in raw_stroke:
            if not isinstance(point, list) or len(point) != 2 or not all(
                type(value) in (int, float) and isfinite(value) for value in point
            ):
                raise ValueError("Некорректные координаты штриха")
            stroke.append((float(point[0]), float(point[1])))
        strokes.append(stroke)
    normalize(strokes)
    return {"label": label, "author": author, "strokes": strokes}


def load_samples(path: Path) -> list[Sample]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    if not isinstance(data, list):
        raise ValueError("Ожидался список образцов")
    return [_parse_sample(item) for item in data]


def append_sample(path: Path, sample: Sample) -> list[Sample]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(path) + ".lock")
    if not lock.tryLock(0):
        raise OSError("Файл образцов занят. Повторите попытку.")
    try:
        samples = load_samples(path)
        samples.append(sample)
        payload = json.dumps(samples, ensure_ascii=False, indent=2).encode("utf-8")
        file = QSaveFile(str(path))
        if not file.open(QIODevice.OpenModeFlag.WriteOnly):
            raise OSError(file.errorString())
        if file.write(payload) != len(payload):
            file.cancelWriting()
            raise OSError(file.errorString())
        if not file.commit():
            raise OSError(file.errorString())
        return samples
    finally:
        lock.unlock()

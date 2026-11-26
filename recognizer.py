"""Rasterize handwriting and classify it with the bundled convolutional network."""

from collections.abc import Sequence
from pathlib import Path
from string import ascii_lowercase, ascii_uppercase, digits

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter, QPen, QPolygonF

type Point = tuple[float, float]
type Strokes = Sequence[Sequence[Point]]

DIGIT_MODEL = Path(__file__).with_name("digits.npz")
MODEL = Path(__file__).with_name("symbols.npz")
PUNCTUATION = """.,;:?!'"()%-@$<>"""
SYMBOL_MIN_SPAN = 60
SYMBOLS = digits + ascii_uppercase + ascii_lowercase + PUNCTUATION
GROUPS = {"Все символы": SYMBOLS, "Цифры": digits, "A–Z": ascii_uppercase,
          "a–z": ascii_lowercase, "Знаки": PUNCTUATION}


def draw_stroke(painter: QPainter, stroke: Sequence[Point] | np.ndarray) -> None:
    line = QPolygonF([QPointF(x, y) for x, y in stroke])
    if len(line) == 1:
        painter.drawPoint(line[0])
    else:
        painter.drawPolyline(line)


def normalized_strokes(strokes: Strokes, min_span: float = 0) -> list[np.ndarray]:
    """Center the drawing; preserve small marks relative to the minimum span."""
    paths = [np.asarray(stroke, dtype=float) for stroke in strokes if len(stroke)]
    if not paths:
        raise ValueError("Нарисуйте символ")
    points = np.concatenate(paths)
    if not np.isfinite(points).all():
        raise ValueError("Некорректные координаты штриха")
    if not min_span and not any(np.any(path != path[0]) for path in paths):
        raise ValueError("Нарисуйте цифру")
    low = points.min(axis=0)
    span = points.max(axis=0) - low
    scale = max(span.max(), min_span)
    if not np.isfinite(scale):
        raise ValueError("Некорректный размер рисунка")
    center = 0.5 * (span / scale)
    return [(path - low) / scale - center for path in paths]


def rasterize(strokes: Strokes, *, min_span: float = 0) -> np.ndarray:
    """Fit the ink into a centered 26-pixel square on a 32-pixel canvas."""
    paths = normalized_strokes(strokes, min_span)

    # Supersample solid ink. Overlapping strokes form a union.
    image = QImage(128, 128, QImage.Format.Format_Grayscale8)
    image.fill(Qt.GlobalColor.black)
    pen = QPen(Qt.GlobalColor.white, 8, Qt.PenStyle.SolidLine,
               Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    with QPainter(image) as painter:
        painter.setPen(pen)
        for path in paths:
            if min_span and np.all(path == path[0]):
                path = path[:1]
            draw_stroke(painter, path * 104 + 64)
    pixels = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(32, 4, 32, 4)
    return pixels.mean(axis=(1, 3), dtype=np.float32) / 255


def _conv_pool(image, weight, bias):
    windows = sliding_window_view(image, (3, 3), axis=(1, 2))
    features = np.einsum("chwkl,ockl->ohw", windows, weight, optimize=True)
    features = np.maximum(features + bias[:, None, None], 0)
    channels, height, width = features.shape
    h, w = height // 2, width // 2
    return features[:, :2*h, :2*w].reshape(channels, h, 2, w, 2).max(axis=(2, 4))


class Recognizer:
    def __init__(self) -> None:
        with np.load(MODEL, allow_pickle=False) as archive:
            self.weights = dict(archive)
        with np.load(DIGIT_MODEL, allow_pickle=False) as archive:
            self.digit_weights = dict(archive)

    @staticmethod
    def _probabilities(image: np.ndarray, weights, alphabet: str) -> dict[str, float]:
        image = image[None]
        for layer in ("conv1", "conv2"):
            image = _conv_pool(image, weights[layer + ".weight"], weights[layer + ".bias"])
        features = np.maximum(weights["embed.weight"] @ image.ravel() + weights["embed.bias"], 0)
        scores = weights["head.weight"] @ features + weights["head.bias"]
        labels = weights["labels"]
        selected = [i for i, label in enumerate(labels) if label in alphabet]
        if not selected:
            raise ValueError("В группе нет поддерживаемых символов")
        scores = scores[selected]
        relative = np.exp((scores - scores.max()).astype(np.float64))
        probabilities = relative / relative.sum()
        return {str(label): float(p) for label, p in zip(labels[selected], probabilities)}

    def predict_probabilities(self, strokes: Strokes, alphabet: str = SYMBOLS) -> dict[str, float]:
        if alphabet == digits:
            return self._probabilities(rasterize(strokes), self.digit_weights, digits)
        image = rasterize(strokes, min_span=SYMBOL_MIN_SPAN)
        probabilities = self._probabilities(image, self.weights, alphabet)
        if alphabet == SYMBOLS:
            # The combined model estimates P(digit). The specialist supplies P(class | digit).
            digit_mass = sum(probabilities[digit] for digit in digits)
            digit_probs = self._probabilities(image, self.digit_weights, digits)
            probabilities.update({digit: digit_mass * p for digit, p in digit_probs.items()})
        return probabilities

    def predict(self, strokes: Strokes, alphabet: str = SYMBOLS) -> str:
        probabilities = self.predict_probabilities(strokes, alphabet)
        return max(probabilities, key=probabilities.__getitem__)

import sys
from pathlib import Path

from PySide6.QtCore import QPointF, QStandardPaths, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPaintEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from recognizer import Point, Recognizer, normalize
from samples import DEMO, Sample, append_sample, load_samples


class DrawingCanvas(QWidget):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.strokes: list[list[Point]] = []
        self._active = False
        self.setMinimumSize(360, 340)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setAccessibleName("Холст для рисования цифры")

    def clear(self) -> None:
        self.strokes.clear()
        self._active = False
        self.changed.emit()
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            position = event.position()
            self.strokes.append([(position.x(), position.y())])
            self._active = True
            self.changed.emit()
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._active and event.buttons() & Qt.MouseButton.LeftButton:
            self._append_point(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._active and event.button() == Qt.MouseButton.LeftButton:
            self._append_point(event.position())
            self._active = False

    def _append_point(self, position: QPointF) -> None:
        point = (position.x(), position.y())
        if point != self.strokes[-1][-1]:
            self.strokes[-1].append(point)
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        with QPainter(self) as painter:
            painter.fillRect(self.rect(), QColor("#ffffff"))
            painter.setPen(QColor("#cbd5e1"))
            painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(QColor("#17233c"), 4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            for stroke in self.strokes:
                points = [QPointF(*point) for point in stroke]
                if len(points) == 1:
                    painter.drawPoint(points[0])
                else:
                    painter.drawPolyline(points)


class MainWindow(QWidget):
    def __init__(self, sample_path: Path) -> None:
        super().__init__()
        self.sample_path = sample_path
        self.recognizer = Recognizer()
        self.demo_templates = [
            (label, normalize(strokes))
            for label, variants in DEMO.items()
            for strokes in variants
        ]
        self.setWindowTitle("Распознавание рукописных цифр")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        layout.addWidget(QLabel("Нарисуйте одну цифру от 0 до 9 мышью."))
        self.canvas = DrawingCanvas()
        layout.addWidget(self.canvas, 1)

        actions = QHBoxLayout()
        recognize = QPushButton("Распознать")
        clear = QPushButton("Очистить")
        actions.addWidget(recognize)
        actions.addWidget(clear)
        actions.addStretch()
        self.result = QLabel()
        actions.addWidget(self.result)
        layout.addLayout(actions)

        sample_form = QHBoxLayout()
        digit_caption = QLabel("Правильная цифра")
        self.digit = QComboBox()
        self.digit.addItems(list(DEMO))
        digit_caption.setBuddy(self.digit)
        author_caption = QLabel("Автор")
        self.author = QLineEdit("local")
        author_caption.setBuddy(self.author)
        save = QPushButton("Сохранить образец")
        for widget in (digit_caption, self.digit, author_caption, self.author, save):
            sample_form.addWidget(widget)
        layout.addLayout(sample_form)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.storage_warning = QLabel()
        self.storage_warning.setWordWrap(True)
        self.storage_warning.hide()
        layout.addWidget(self.storage_warning)

        recognize.clicked.connect(self.recognize)
        clear.clicked.connect(self.canvas.clear)
        save.clicked.connect(self.save_sample)
        self.canvas.changed.connect(self.result.clear)
        self.canvas.changed.connect(self.status.clear)

        try:
            samples = load_samples(self.sample_path)
        except (OSError, ValueError, OverflowError) as error:
            samples = []
            self.storage_warning.setText(f"Не удалось загрузить свои образцы. {error}")
            self.storage_warning.show()
        self._set_templates(samples)
        self.resize(620, 540)

    def _set_templates(self, samples: list[Sample]) -> None:
        self.recognizer.templates = self.demo_templates + [
            (sample["label"], normalize(sample["strokes"])) for sample in samples
        ]

    def recognize(self) -> None:
        try:
            label = self.recognizer.predict(self.canvas.strokes)
        except ValueError as error:
            self.result.clear()
            self.status.setText(str(error))
            return
        self.status.clear()
        self.result.setText(f"Результат: {label}")

    def save_sample(self) -> None:
        try:
            normalize(self.canvas.strokes)
        except ValueError as error:
            self.status.setText(str(error))
            return
        author = self.author.text().strip()
        if not author:
            self.status.setText("Укажите автора образца")
            return

        sample: Sample = {
            "label": self.digit.currentText(),
            "author": author,
            "strokes": [stroke.copy() for stroke in self.canvas.strokes],
        }
        try:
            samples = append_sample(self.sample_path, sample)
        except (OSError, ValueError, OverflowError) as error:
            self.status.setText(f"Не удалось сохранить образец. {error}")
            return
        self._set_templates(samples)
        self.storage_warning.hide()
        self.result.clear()
        self.status.setText(f"Образец цифры {sample['label']} сохранён")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SymbolOnLine")
    data_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    window = MainWindow(Path(data_dir) / "samples.json")
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

import sys

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPaintEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from recognizer import Point, Recognizer, draw_stroke


WORKSPACE_STYLE = """
QWidget#workspace { background: #f5f7fa; }
QLabel { color: #1b2536; }
QLabel#brand { font-size: 14px; font-weight: 600; }
QLabel#heading { font-size: 23px; font-weight: 600; }
QLabel#instruction, QLabel#status { color: #526078; }
QWidget#result-panel {
    background: #edf2ff; border: 1px solid #d6dee9; border-radius: 12px;
}
QLabel#result-label { color: #3d4e70; font-weight: 600; }
QLabel#result { color: #264fc9; font-size: 48px; font-weight: 600; }
QPushButton {
    color: #1b2536; background: white; border: 1px solid #c4cfdf;
    border-radius: 7px; padding: 8px 16px; font-weight: 600;
}
QPushButton:hover { background: #e8eef9; }
QPushButton:pressed { background: #d6e2f8; }
QPushButton#recognize { color: white; background: #264fc9; border-color: #264fc9; }
QPushButton#recognize:hover { background: #1f43ad; }
QPushButton#recognize:pressed { background: #1b378c; }
QPushButton:focus, QPushButton#recognize:focus { border-color: #121e39; }
"""


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
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            outline = QPainterPath()
            outline.addRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)
            painter.fillPath(outline, QColor("#ffffff"))
            painter.setClipPath(outline)
            painter.setPen(QColor("#d6dfea"))
            for y in range(8, self.height(), 16):
                for x in range(8, self.width(), 16):
                    painter.drawPoint(x, y)

            pen = QPen(QColor("#17233c"), 4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            for stroke in self.strokes:
                draw_stroke(painter, stroke)
            painter.setClipping(False)
            painter.strokePath(outline, QPen(QColor("#d6dee9")))


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("workspace")
        self.setWindowTitle("SymbolOnLine — Распознавание цифр")
        self.setStyleSheet(WORKSPACE_STYLE)
        self.recognizer = Recognizer()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 22)
        layout.setSpacing(16)
        brand = QLabel("SymbolOnLine")
        brand.setObjectName("brand")
        layout.addWidget(brand)

        body = QHBoxLayout()
        body.setSpacing(20)
        layout.addLayout(body, 1)
        drawing = QVBoxLayout()
        drawing.setSpacing(10)
        body.addLayout(drawing, 1)
        heading = QLabel("Распознавание цифр")
        heading.setObjectName("heading")
        drawing.addWidget(heading)
        instruction = QLabel("Нарисуйте одну цифру от 0 до 9 мышью.")
        instruction.setObjectName("instruction")
        drawing.addWidget(instruction)
        self.canvas = DrawingCanvas()
        drawing.addWidget(self.canvas, 1)

        actions = QHBoxLayout()
        recognize = QPushButton("Распознать")
        recognize.setObjectName("recognize")
        clear = QPushButton("Очистить")
        actions.addWidget(recognize)
        actions.addWidget(clear)
        actions.addStretch()
        drawing.addLayout(actions)

        sidebar = QVBoxLayout()
        sidebar.setSpacing(10)
        body.addLayout(sidebar)
        panel = QWidget()
        panel.setObjectName("result-panel")
        panel.setMinimumWidth(168)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 14, 16, 14)
        panel_layout.setSpacing(2)
        result_label = QLabel("Результат")
        result_label.setObjectName("result-label")
        panel_layout.addWidget(result_label)
        self.result = QLabel("—")
        self.result.setObjectName("result")
        self.result.setAccessibleName("Распознанная цифра")
        self.result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel_layout.addWidget(self.result)
        sidebar.addWidget(panel)

        self.status = QLabel()
        self.status.setObjectName("status")
        self.status.setFixedWidth(168)
        self.status.setWordWrap(True)
        sidebar.addWidget(self.status)
        sidebar.addStretch()

        recognize.clicked.connect(self.recognize)
        clear.clicked.connect(self.canvas.clear)
        self.canvas.changed.connect(self.reset_result)

        self.resize(800, 535)

    def reset_result(self) -> None:
        self.result.setText("—")
        self.status.clear()

    def recognize(self) -> None:
        try:
            label = self.recognizer.predict(self.canvas.strokes)
        except ValueError as error:
            self.result.setText("—")
            self.status.setText(str(error))
            return
        self.status.clear()
        self.result.setText(label)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SymbolOnLine")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

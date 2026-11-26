import sys

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPaintEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from recognizer import GROUPS, Point, Recognizer, draw_stroke


WORKSPACE_STYLE = """
QWidget#workspace { background: #f5f7fa; }
QLabel { color: #1b2536; }
QLabel#heading { font-size: 23px; font-weight: 600; }
QLabel#instruction, QLabel#status { color: #526078; }
QWidget#result-panel {
    background: #edf2ff; border: 1px solid #d6dee9; border-radius: 12px;
}
QWidget#probability-panel { background: white; border: 1px solid #d6dee9; border-radius: 12px; }
QLabel#result-label, QLabel#probability-label { color: #3d4e70; font-weight: 600; }
QLabel#probability-label { font-size: 15px; padding-bottom: 4px; }
QLabel#probability-digit {
    background: #edf2ff; color: #264fc9; border-radius: 6px;
    min-width: 26px; padding: 2px 0; font-size: 16px; font-weight: 600;
}
QLabel#probability-value { color: #263b61; font-size: 16px; font-weight: 600; }
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
        self.clear_on_next_stroke = False
        self.setMinimumSize(360, 340)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setAccessibleName("Холст для рисования символа")

    def clear(self) -> None:
        self.strokes.clear()
        self._active = False
        self.clear_on_next_stroke = False
        self.changed.emit()
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self.clear_on_next_stroke:
                self.strokes.clear()
                self.clear_on_next_stroke = False
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
        self.setWindowTitle("Распознавание символов")
        self.setStyleSheet(WORKSPACE_STYLE)
        self.recognizer = Recognizer()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 22)
        layout.setSpacing(10)
        heading = QLabel("Распознавание символов")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        options = QHBoxLayout()
        instruction = QLabel("Нарисуйте букву, цифру или знак. Для точки щёлкните мышью.")
        instruction.setObjectName("instruction")
        instruction.setWordWrap(True)
        options.addWidget(instruction, 1)
        self.group = QComboBox()
        self.group.setAccessibleName("Группа символов")
        for name, alphabet in GROUPS.items():
            self.group.addItem(name, alphabet)
        options.addWidget(self.group)
        layout.addLayout(options)

        body = QHBoxLayout()
        body.setSpacing(20)
        layout.addLayout(body, 1)
        drawing = QVBoxLayout()
        drawing.setSpacing(10)
        body.addLayout(drawing, 1)
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
        self.result.setAccessibleName("Распознанный символ")
        self.result.setTextFormat(Qt.TextFormat.PlainText)
        self.result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel_layout.addWidget(self.result)
        sidebar.addWidget(panel)

        self.probability_panel = QWidget()
        self.probability_panel.setObjectName("probability-panel")
        probability_layout = QVBoxLayout(self.probability_panel)
        probability_layout.setContentsMargins(16, 14, 16, 14)
        heading = QLabel("Вероятности")
        heading.setObjectName("probability-label")
        probability_layout.addWidget(heading)
        rows = QWidget()
        self.probability_grid = QGridLayout(rows)
        self.probability_grid.setContentsMargins(0, 0, 0, 0)
        self.probability_grid.setVerticalSpacing(6)
        self.probability_grid.setColumnStretch(1, 1)
        self.probability_scroll = QScrollArea()
        self.probability_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.probability_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.probability_scroll.setWidgetResizable(True)
        self.probability_scroll.setWidget(rows)
        probability_layout.addWidget(self.probability_scroll)
        self.probability_rows = {}
        sidebar.addWidget(self.probability_panel)

        self.status = QLabel()
        self.status.setObjectName("status")
        self.status.setFixedWidth(168)
        self.status.setWordWrap(True)
        sidebar.addWidget(self.status)
        sidebar.addStretch()

        recognize.clicked.connect(self.recognize)
        clear.clicked.connect(self.canvas.clear)
        self.canvas.changed.connect(self.reset_result)
        self.group.currentIndexChanged.connect(self.group_changed)

        self.reset_result()
        self.resize(800, 535)

    def reset_result(self) -> None:
        self.result.setText("—")
        self.probability_panel.hide()
        self.status.clear()

    def group_changed(self) -> None:
        if self.canvas.clear_on_next_stroke:
            self.recognize()
        else:
            self.reset_result()

    def recognize(self) -> None:
        try:
            probabilities = self.recognizer.predict_probabilities(self.canvas.strokes, self.group.currentData())
        except ValueError as error:
            self.reset_result()
            self.status.setText(str(error))
            return
        self.status.clear()
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        self.result.setText(ranked[0][0])
        for symbol, value in self.probability_rows.values():
            symbol.hide()
            value.hide()
        for row, (label, probability) in enumerate(ranked):
            if probability <= 0.01:
                break
            if label not in self.probability_rows:
                symbol = QLabel(label)
                symbol.setTextFormat(Qt.TextFormat.PlainText)
                symbol.setObjectName("probability-digit")
                symbol.setAlignment(Qt.AlignmentFlag.AlignCenter)
                value = QLabel()
                value.setObjectName("probability-value")
                value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                value.setAccessibleName(f"Вероятность символа {label}")
                self.probability_rows[label] = symbol, value
            symbol, value = self.probability_rows[label]
            self.probability_grid.addWidget(symbol, row, 0)
            self.probability_grid.addWidget(value, row, 1)
            percent = f"{probability * 100:.2f}".replace(".", ",")
            value.setText(f"{percent} %")
            symbol.show()
            value.show()
        self.probability_scroll.setFixedHeight(min(self.probability_grid.sizeHint().height(), 220))
        self.probability_scroll.verticalScrollBar().setValue(0)
        self.probability_panel.show()
        self.canvas.clear_on_next_stroke = True


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SymbolOnLine")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

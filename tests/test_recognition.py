"""Mouse paths traced from the five reported screenshots, in canvas coordinates."""

import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from main import MainWindow
from recognizer import Recognizer, rasterize
from samples import DEMO

EXAMPLES = {
    "2": [[(233,311),(235,288),(237,259),(238,249),(243,239),(253,234),(265,232),
           (277,233),(285,241),(293,256),(301,276),(305,291),(305,317),(301,329),
           (293,340),(282,350),(268,358),(245,364),(270,370),(295,373),(357,373)]],
    "5": [[(267,284),(284,279),(312,275),(348,272)],
          [(270,291),(266,305),(270,309),(294,314),(309,321),(318,331),(322,343),
           (322,352),(319,358),(313,359),(302,352),(285,343)]],
    "7": [[(245,299),(265,295),(304,295),(318,298),(328,304),(332,315),(332,344),
           (330,369),(329,416),(330,436),(334,454)], [(289,385),(320,377),(353,368)]],
    "8": [[(268,296),(268,291),(259,287),(231,287),(217,292),(206,298),(203,305),
           (204,317),(212,331),(228,347),(248,364),(273,382),(285,393),(289,402),
           (288,407),(282,410),(260,409),(241,405),(231,400),(229,391),(230,383),
           (239,366),(251,348),(262,330),(272,307),(278,297),(279,293)]],
    "9": [[(310,288),(308,297),(303,299),(293,298),(284,294),(275,284),(267,268),
           (264,257),(268,252),(286,252),(297,254),(304,261),(309,275),(310,292),
           (314,308),(319,325),(320,340),(314,343),(305,344),(294,339),(282,336)]],
}


class RecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.model = Recognizer()

    def test_reported_and_builtin_shapes(self):
        for label, variants in DEMO.items():
            for strokes in variants:
                with self.subTest(label=label, strokes=strokes):
                    self.assertEqual(self.model.predict(strokes), label)
        for label, strokes in EXAMPLES.items():
            with self.subTest(label=label):
                self.assertEqual(self.model.predict(strokes), label)
                transformed = [[(2*x + 11, 2*y - 25) for x, y in reversed(path)]
                               for path in reversed(strokes)]
                self.assertEqual(self.model.predict(transformed), label)
                np.testing.assert_array_equal(rasterize(strokes), rasterize(transformed))

    def test_invalid_input(self):
        for drawing in ([], [[(10, 20)]], [[(1, 1), (1, 1)]], [[(0, 0), (np.nan, 2)]]):
            with self.assertRaises(ValueError):
                self.model.predict(drawing)

    def test_window_recognize_and_clear(self):
        window = MainWindow()
        window.show()
        canvas = window.canvas
        for stroke in EXAMPLES["7"]:
            points = [QPoint(x - 210, y - 229) for x, y in stroke]
            QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=points[0])
            for point in points[1:-1]:
                QTest.mouseMove(canvas, point)
            QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=points[-1])
        recognize = window.findChild(QPushButton, "recognize")
        QTest.mouseClick(recognize, Qt.MouseButton.LeftButton)
        self.assertEqual(window.result.text(), "7")
        probabilities = self.model.predict_probabilities(canvas.strokes)
        self.assertEqual(set(probabilities), set("0123456789"))
        self.assertAlmostEqual(sum(probabilities.values()), 1)
        for label, probability in probabilities.items():
            digit_label, value = window.probability_rows[label]
            self.assertEqual(digit_label.isVisible(), probability > 0.01)
            self.assertEqual(value.isVisible(), probability > 0.01)
            if value.isVisible():
                shown = value.text().removesuffix(" %").replace(",", ".")
                self.assertAlmostEqual(float(shown), 100 * probability, delta=0.005)
        clear = next(button for button in window.findChildren(QPushButton)
                     if button.text() == "Очистить")
        QTest.mouseClick(clear, Qt.MouseButton.LeftButton)
        self.assertEqual(canvas.strokes, [])
        self.assertEqual(window.result.text(), "—")
        self.assertTrue(window.probability_panel.isHidden())
        QTest.mouseClick(recognize, Qt.MouseButton.LeftButton)
        self.assertEqual(window.status.text(), "Нарисуйте цифру")
        window.close()

    def test_first_stroke_after_recognition_replaces_drawing(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window.show()
        canvas = window.canvas
        left = Qt.MouseButton.LeftButton
        QTest.mousePress(canvas, left, pos=QPoint(10, 10))
        QTest.mouseRelease(canvas, left, pos=QPoint(10, 90))
        QTest.mouseClick(window.findChild(QPushButton, "recognize"), left)
        self.assertEqual(canvas.strokes, [[(10, 10), (10, 90)]])
        self.assertTrue(window.probability_panel.isVisible())

        QTest.mousePress(canvas, left, pos=QPoint(20, 30))
        self.assertEqual(canvas.strokes, [[(20, 30)]])
        self.assertEqual(window.result.text(), "—")
        self.assertTrue(window.probability_panel.isHidden())
        QTest.mouseRelease(canvas, left, pos=QPoint(60, 30))
        QTest.mousePress(canvas, left, pos=QPoint(60, 30))
        QTest.mouseRelease(canvas, left, pos=QPoint(40, 90))
        self.assertEqual(canvas.strokes, [[(20, 30), (60, 30)], [(60, 30), (40, 90)]])

    def test_probability_cutoff_before_rounding(self):
        window = MainWindow()
        window.show()
        probabilities = dict.fromkeys("0123456789", 0.0)
        probabilities.update({"0": 0.01, "1": 0.01001, "2": 0.00999, "3": 0.97})
        with patch.object(window.recognizer, "predict_probabilities", return_value=probabilities):
            window.recognize()
        self.app.processEvents()
        visible = [digit for digit, _ in window.probability_rows.values() if digit.isVisible()]
        self.assertEqual([digit.text() for digit in sorted(visible, key=lambda w: w.y())], ["3", "1"])
        result_panel = window.result.parentWidget()
        self.assertEqual(result_panel.y(), window.canvas.y())
        self.assertGreater(window.probability_panel.y(), result_panel.geometry().bottom())
        self.assertEqual(window.result.text(), "3")
        window.close()

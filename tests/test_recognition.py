import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from main import MainWindow
from recognizer import GROUPS, SYMBOLS, WRITING_GUIDES, Recognizer, guide_features, rasterize
from samples import DEMO


class RecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.model = Recognizer()

    def test_builtin_digit_shapes(self):
        for label, variants in DEMO.items():
            for strokes in variants:
                with self.subTest(label=label, strokes=strokes):
                    self.assertEqual(self.model.predict(strokes, GROUPS["Цифры"]), label)
                    transformed = [[(2*x + 11, 2*y - 25) for x, y in reversed(path)]
                                   for path in reversed(strokes)]
                    self.assertEqual(self.model.predict(transformed, GROUPS["Цифры"]), label)
                    np.testing.assert_array_equal(rasterize(strokes), rasterize(transformed))

    def test_invalid_input(self):
        for drawing in ([], [[]], [[(0, 0), (np.nan, 2)]]):
            with self.assertRaises(ValueError):
                self.model.predict(drawing)

    def test_geometry_and_position_independent_modes(self):
        upper, baseline = WRITING_GUIDES
        strokes = [[(15, upper), (50, baseline)], [(20, 160), (40, 160)]]
        np.testing.assert_allclose(guide_features(strokes, WRITING_GUIDES), [1, 0])
        digits = GROUPS["Цифры"]
        self.assertEqual(self.model.predict_probabilities(strokes, digits),
                         self.model.predict_probabilities(strokes, digits, guides=WRITING_GUIDES))
        for alphabet in (SYMBOLS, GROUPS["Знаки"]):
            for y in (10, baseline, 330):
                self.assertEqual(self.model.predict([[(80, y)]], alphabet, guides=WRITING_GUIDES), ".")

    def test_guides_toggle_and_resize_on_existing_ink(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window.show()
        canvas = window.canvas
        self.assertTrue(window.guide_toggle.isChecked())
        points = [QPoint(x, y + 70) for x, y in [(72, 0), (74, 4), (73, 10), (68, 19), (61, 24)]]
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=points[0])
        for point in points[1:-1]:
            QTest.mouseMove(canvas, point)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=points[-1])
        QTest.mouseClick(window.findChild(QPushButton, "recognize"), Qt.MouseButton.LeftButton)
        self.assertEqual(window.result.text(), "'")
        window.resize(960, 640)
        self.app.processEvents()
        guided_pixel = canvas.grab().toImage().pixelColor(100, 203)
        window.guide_toggle.click()
        self.assertNotEqual(canvas.grab().toImage().pixelColor(100, 203), guided_pixel)
        self.assertEqual(window.result.text(), ",")
        window.guide_toggle.click()
        self.assertEqual(window.result.text(), "'")
        self.assertEqual(canvas.guides, WRITING_GUIDES)
        self.assertEqual(canvas.strokes, [[(p.x(), p.y()) for p in points]])
        self.assertTrue(canvas.clear_on_next_stroke)

    def test_auto_symbols_and_group_probabilities(self):
        drawings = {
            ".": [[(90, 40)]],
            "A": [[(10, 100), (45, 10), (80, 100)], [(24, 65), (66, 65)]],
            "i": [[(35, 48), (35, 100)], [(35, 23)]],
            ":": [[(45, 30)], [(45, 80)]],
            ",": [[(10, 10), (13, 12), (14, 17), (11, 23), (4, 31)]],
        }
        for label, strokes in drawings.items():
            with self.subTest(label=label):
                probabilities = self.model.predict_probabilities(strokes)
                self.assertEqual(set(probabilities), set(SYMBOLS))
                self.assertAlmostEqual(sum(probabilities.values()), 1)
                self.assertEqual(max(probabilities, key=probabilities.get), label)
        for alphabet in GROUPS.values():
            with self.subTest(alphabet=alphabet):
                probabilities = self.model.predict_probabilities(drawings["A"], alphabet)
                self.assertEqual(set(probabilities), set(alphabet))
                self.assertAlmostEqual(sum(probabilities.values()), 1)

    def test_window_recognize_and_clear(self):
        window = MainWindow()
        window.show()
        window.group.setCurrentText("Цифры")
        canvas = window.canvas
        for stroke in DEMO["7"][1]:
            points = [QPoint(20 * x, 20 * y) for x, y in stroke]
            QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=points[0])
            for point in points[1:-1]:
                QTest.mouseMove(canvas, point)
            QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=points[-1])
        recognize = window.findChild(QPushButton, "recognize")
        QTest.mouseClick(recognize, Qt.MouseButton.LeftButton)
        self.assertEqual(window.result.text(), "7")
        probabilities = self.model.predict_probabilities(canvas.strokes, GROUPS["Цифры"])
        self.assertEqual(set(probabilities), set("0123456789"))
        self.assertAlmostEqual(sum(probabilities.values()), 1)
        for label, probability in probabilities.items():
            if probability <= 0.01:
                self.assertNotIn(label, window.probability_rows)
                continue
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
        self.assertEqual(window.status.text(), "Нарисуйте символ")
        window.close()

    def test_click_is_a_period_in_window(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window.show()
        QTest.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(80, 80))
        QTest.mouseClick(window.findChild(QPushButton, "recognize"), Qt.MouseButton.LeftButton)
        self.assertEqual(window.result.text(), ".")
        self.assertTrue(window.probability_rows["."][0].isVisible())
        window.group.setCurrentText("Знаки")
        self.assertEqual(window.result.text(), ".")

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

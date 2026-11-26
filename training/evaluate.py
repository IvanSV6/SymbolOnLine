"""Compare the shipped classifier and the historical $P implementation on UNIPEN."""
import argparse
import subprocess
from types import ModuleType

import numpy as np

from recognizer import Recognizer
from samples import DEMO
from training.pendigits import records


def baseline():
    source = subprocess.check_output(["git", "show", "9acb4ab:recognizer.py"])
    old = ModuleType("baseline")
    exec(compile(source, "9acb4ab:recognizer.py", "exec"), old.__dict__)
    model = old.Recognizer()
    model.templates = [(label, old.normalize(strokes))
                       for label, variants in DEMO.items() for strokes in variants]
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    model = baseline() if args.baseline else Recognizer()
    matrix = np.zeros((10, 10), dtype=int)
    for _, label, strokes in records(args.archive, "tes"):
        matrix[label, int(model.predict(strokes))] += 1
    print(matrix)
    print("Accuracy:", matrix.trace() / matrix.sum())
    print("Recall:", matrix.diagonal() / matrix.sum(axis=1))

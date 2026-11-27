"""Compare the shipped classifier and the historical $P implementation on UNIPEN."""
import argparse
from collections import Counter
import subprocess
from string import digits
from types import ModuleType

import numpy as np

from recognizer import GROUPS, SYMBOLS, WRITING_GUIDES, Recognizer
from samples import DEMO
from training.pendigits import records
from training import uji


def baseline():
    source = subprocess.check_output(["git", "show", "9acb4ab:recognizer.py"])
    old = ModuleType("baseline")
    exec(compile(source, "9acb4ab:recognizer.py", "exec"), old.__dict__)
    model = old.Recognizer()
    model.templates = [(label, old.normalize(strokes))
                       for label, variants in DEMO.items() for strokes in variants]
    return model


def compare_guides(archive):
    model = Recognizer()
    totals = Counter()
    correct = [Counter(), Counter()]
    case_errors = [0, 0]
    for label, strokes in uji.records(archive, "tst"):
        totals[label] += 1
        for i, guides in enumerate((None, WRITING_GUIDES)):
            pred = model.predict(strokes, guides=guides)
            correct[i][label] += pred == label
            case_errors[i] += pred != label and label.isalpha() and pred.lower() == label.lower()
    print("Class / count / correct (free) / correct (guided)")
    for label in "CcZzOo0,'":
        print(label, totals[label], correct[0][label], correct[1][label])
    print("Total", totals.total(), correct[0].total(), correct[1].total())
    print("Case errors:", *case_errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--uji", action="store_true", help="Evaluate all supported ASCII characters on UJI test writers")
    parser.add_argument("--compare-guides", action="store_true", help="Compare free and guided writing on UJI test writers")
    args = parser.parse_args()
    if args.compare_guides:
        if args.baseline:
            parser.error("The baseline only supports Pendigits")
        compare_guides(args.archive)
        raise SystemExit(0)
    if args.uji:
        if args.baseline:
            parser.error("The baseline only supports Pendigits")
        model = Recognizer()
        totals, automatic, restricted = Counter(), Counter(), Counter()
        for label, strokes in uji.records(args.archive, "tst"):
            group = next(name for name, alphabet in GROUPS.items() if alphabet != SYMBOLS and label in alphabet)
            totals[group] += 1
            automatic[group] += model.predict(strokes) == label
            restricted[group] += model.predict(strokes, GROUPS[group]) == label
        print("Group / count / correct (auto) / correct (group)")
        for group, count in totals.items():
            print(group, count, automatic[group], restricted[group])
        print("Total", totals.total(), automatic.total(), restricted.total())
        raise SystemExit(0)
    model = baseline() if args.baseline else Recognizer()
    matrix = np.zeros((10, 10), dtype=int)
    for _, label, strokes in records(args.archive, "tes"):
        prediction = model.predict(strokes) if args.baseline else model.predict(strokes, digits)
        matrix[label, int(prediction)] += 1
    print(matrix)
    print("Accuracy:", matrix.trace() / matrix.sum())
    print("Recall:", matrix.diagonal() / matrix.sum(axis=1))

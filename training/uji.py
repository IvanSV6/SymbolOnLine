"""Read ASCII characters from the UJIpenchars2 writer-independent split."""

from pathlib import Path
from zipfile import ZipFile


def records(archive: Path, split: str):
    with ZipFile(archive) as source:
        lines = iter(source.read("ujipenchars2.txt").decode("utf-8").splitlines())
    for line in lines:
        if not line.startswith("WORD "):
            continue
        _, label, session = line.split()
        count = int(next(lines).split()[1])
        # Convert tablet ink units to canvas pixels at 16 px/mm.
        units_per_mm = 152 if "_UPV_" in session else 100
        strokes = []
        for _ in range(count):
            fields = next(lines).split("#", 1)[1].split()
            coords = [int(value) * 16 / units_per_mm for value in fields]
            strokes.append(list(zip(coords[::2], coords[1::2])))
        if label.isascii() and session.startswith(split + "_"):
            yield label, strokes

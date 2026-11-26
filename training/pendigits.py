"""Read the original UNIPEN trajectories (gzip -dc also reads Unix compress)."""

from collections.abc import Iterator
from pathlib import Path
import subprocess
import zipfile

from recognizer import Strokes


def records(archive: Path, split: str) -> Iterator[tuple[int, int, Strokes]]:
    with zipfile.ZipFile(archive) as source:
        packed = source.read(f"pendigits-orig.{split}.Z")
    lines = subprocess.check_output(["gzip", "-dc"], input=packed).decode("ascii").splitlines()
    strokes = []
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == ".SEGMENT":
            if strokes:
                yield writer, label, strokes
            strokes = []
            label = int(fields[-1].strip('"'))
        elif fields[0] == ".COMMENT":
            writer = int(fields[2])
        elif fields[0] == ".PEN_DOWN":
            strokes.append([])
        elif not fields[0].startswith("."):
            x, y = map(int, fields)
            strokes[-1].append((x, -y))  # Tablet Y increases upwards.
    if strokes:
        yield writer, label, strokes

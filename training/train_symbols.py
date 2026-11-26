"""Train the combined ASCII classifier on the training writers from UJI and Pendigits."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from recognizer import MODEL, SYMBOL_MIN_SPAN, SYMBOLS, rasterize
from samples import DEMO
from training import pendigits, uji
from training.train import Network, augment


def train(pen_archive, uji_archive, output):
    torch.manual_seed(189)
    torch.set_num_threads(4)
    data = list(uji.records(uji_archive, "trn"))
    # A mouse tap makes an exact dot. Add printed i variants with a straight stem
    # to complement the hooked, handwritten stems common in the tablet data.
    rng = np.random.default_rng(189)
    for _ in range(120):
        height = rng.uniform(35, 95)
        lean = rng.uniform(-0.12, 0.12) * height
        dot = (rng.uniform(-0.08, 0.08) * height, -height * rng.uniform(0.18, 0.65))
        data.append(("i", [[dot], [(0, 0), (lean, height)]]))
    for label, variants in DEMO.items():
        for strokes in variants:
            # DEMO coordinates use a 10-unit em square.
            drawing = [[(10*x, 10*y) for x, y in path] for path in strokes]
            data.extend([(label, drawing)] * 5)
    fixed_count = len(data)
    data.extend((str(label), strokes) for writer, label, strokes
                in pendigits.records(pen_archive, "tra") if writer <= 25)
    images = np.stack([rasterize(s, min_span=SYMBOL_MIN_SPAN) for _, s in data])
    x = torch.from_numpy(images)[:, None]
    y = torch.tensor([SYMBOLS.index(label) for label, _ in data])
    model = Network(classes=len(SYMBOLS), channels=24, features=160)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    epochs = 48
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.00015)
    for epoch in range(epochs):
        # Rotate the extra digit examples so they do not swamp the letter and mark classes.
        pen = torch.randperm(len(data) - fixed_count)[:1200] + fixed_count
        indices = torch.cat((torch.arange(fixed_count), pen))
        for batch in indices[torch.randperm(len(indices))].split(128):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(augment(x[batch])), y[batch])
            loss.backward()
            optimizer.step()
        schedule.step()
        if (epoch + 1) % 8 == 0:
            print(f"Epoch {epoch + 1}/{epochs}", flush=True)
    weights = {key: tensor.numpy() for key, tensor in model.state_dict().items()}
    np.savez_compressed(output, **weights, labels=list(SYMBOLS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pendigits", type=Path)
    parser.add_argument("uji", type=Path)
    parser.add_argument("--output", type=Path, default=MODEL)
    args = parser.parse_args()
    train(args.pendigits, args.uji, args.output)

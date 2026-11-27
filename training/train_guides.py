"""Fit a positional logit correction on 32 UJI writers and select its strength on eight others."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from recognizer import GUIDE_MODEL, SYMBOL_MIN_SPAN, WRITING_GUIDES, Recognizer, guide_features, rasterize
from training.uji import writer_records


def train(archive: Path, output: Path):
    torch.manual_seed(189)
    torch.set_num_threads(2)
    model = Recognizer()
    records = list(writer_records(archive, "trn"))
    writers = sorted({writer for writer, _, _ in records})
    validation_writers = set(np.random.default_rng(189).choice(writers, 8, replace=False))
    validation = [(label, s) for writer, label, s in records if writer in validation_writers]
    train_rows = [(label, s) for writer, label, s in records if writer not in validation_writers]
    labels = list(model.weights["labels"])
    x = torch.from_numpy(np.stack([guide_features(s, WRITING_GUIDES) for _, s in train_rows]))
    y = torch.tensor([labels.index(label) for label, _ in train_rows])
    # Cache frozen shape logits once. Only the small correction is differentiated.
    scores = torch.from_numpy(np.stack([
        model._scores(rasterize(s, min_span=SYMBOL_MIN_SPAN), model.weights) for _, s in train_rows
    ]))
    branch = nn.Sequential()
    branch.add_module("hidden", nn.Linear(2, 24))
    branch.add_module("relu", nn.ReLU())
    branch.add_module("output", nn.Linear(24, len(labels)))
    nn.init.zeros_(branch.output.weight)
    nn.init.zeros_(branch.output.bias)
    optimizer = torch.optim.AdamW(branch.parameters(), lr=0.004)
    for _ in range(100):
        for batch in torch.randperm(len(y)).split(128):
            correction = branch(x[batch])
            loss = nn.functional.cross_entropy(scores[batch] + correction, y[batch])
            loss += 0.001 * correction.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    weights = {name: value.numpy() for name, value in branch.state_dict().items()}
    model.guide_weights = weights
    best_count = -1
    for strength in (0.0, 0.25, 0.5, 1.0):
        weights["strength"] = np.float32(strength)
        count = sum(model.predict(s, guides=WRITING_GUIDES) == label for label, s in validation)
        print(f"Validation {strength}: {count}/{len(validation)}", flush=True)
        if count > best_count:
            best_count, best_strength = count, strength
    weights["strength"] = np.float32(best_strength)
    np.savez_compressed(output, **weights)
    print("Validation writers:", ", ".join(sorted(validation_writers)))
    print("Selected:", best_strength)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uji", type=Path)
    parser.add_argument("--output", type=Path, default=GUIDE_MODEL)
    args = parser.parse_args()
    train(args.uji, args.output)

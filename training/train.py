"""Train and export the CPU model. Run from the repository root as a module."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from recognizer import MODEL, rasterize
from samples import DEMO
from training.pendigits import records


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, 3)
        self.conv2 = nn.Conv2d(12, 24, 3)
        self.embed = nn.Linear(24 * 6 * 6, 64)
        self.head = nn.Linear(64, 10)

    def features(self, x):
        x = F.max_pool2d(self.conv1(x).relu(), 2)
        x = F.max_pool2d(self.conv2(x).relu(), 2)
        return self.embed(x.flatten(1)).relu()

    def forward(self, x):
        return self.head(self.features(x))


def augment(x):
    n = len(x)
    matrix = torch.rand(n, 2, 3) * 2 - 1
    matrix *= torch.tensor([[0.22, 0.34, 0.10], [0.17, 0.20, 0.10]])
    matrix[:, 0, 0] += 1
    matrix[:, 1, 1] += 1
    grid = F.affine_grid(matrix, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False)
    thickness = torch.rand(n, 1, 1, 1) - 0.45
    thick = F.max_pool2d(x, 3, 1, 1)
    thin = -F.max_pool2d(-x, 3, 1, 1)
    return x + thickness.clamp(min=0) * (thick - x) + (-thickness).clamp(min=0) * (thin - x)


@torch.no_grad()
def calibrate(model, x, labels, writers):
    """One sample per label from the same writer, remaining examples are queries."""
    embeddings = model.features(x)
    base = model.head(embeddings).argmax(1)
    features = F.normalize(embeddings)
    thresholds = torch.linspace(0.80, 1.0, 101)
    correct = torch.zeros_like(thresholds)
    for writer in writers.unique():
        indices = torch.where(writers == writer)[0]
        for offset in range(5):
            support = torch.stack([indices[labels[indices] == label][offset] for label in range(10)])
            query = indices[~torch.isin(indices, support)]
            scores, nearest = (features[query] @ features[support].T).max(1)
            predicted = torch.where(scores[:, None] >= thresholds, labels[support[nearest], None],
                                    base[query, None])
            correct += (predicted == labels[query, None]).sum(0)
    return float(thresholds[correct == correct.max()].max())


def train(archive, output):
    torch.manual_seed(189)
    torch.set_num_threads(4)
    data = [(w, label, rasterize(strokes)) for w, label, strokes in records(archive, "tra")]
    # Include the original twenty supported shapes during fitting.
    for label, variants in DEMO.items():
        for strokes in variants:
            data.extend([(0, int(label), rasterize(strokes))] * 5)
    writers = torch.tensor([w for w, _, _ in data])
    labels = torch.tensor([label for _, label, _ in data])
    x = torch.from_numpy(np.stack([image for _, _, image in data]))[:, None]
    fit = torch.where(writers <= 25)[0]
    validation = writers > 25
    model = Network()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 32, eta_min=0.0002)
    for epoch in range(32):
        for indices in fit[torch.randperm(len(fit))].split(128):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(augment(x[indices])), labels[indices])
            loss.backward()
            optimizer.step()
        schedule.step()
        if (epoch + 1) % 8 == 0:
            with torch.no_grad():
                acc = (model(x[validation]).argmax(1) == labels[validation]).float().mean()
            print(f"Epoch {epoch + 1}: validation accuracy {acc:.4f}", flush=True)
    threshold = calibrate(model, x[validation], labels[validation], writers[validation])
    weights = {key: tensor.numpy() for key, tensor in model.state_dict().items()}
    np.savez_compressed(output, **weights, labels=list("0123456789"), threshold=threshold)
    print(f"Saved {output}, personal similarity threshold {threshold:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", default=MODEL, type=Path)
    args = parser.parse_args()
    train(args.archive, args.output)

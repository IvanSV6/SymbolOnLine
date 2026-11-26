"""Train and export the CPU model. Run from the repository root as a module."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from recognizer import DIGIT_MODEL, rasterize
from samples import DEMO
from training.pendigits import records


class Network(nn.Module):
    def __init__(self, classes=10, channels=12, features=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3)
        self.conv2 = nn.Conv2d(channels, 2 * channels, 3)
        self.embed = nn.Linear(2 * channels * 6 * 6, features)
        self.head = nn.Linear(features, classes)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x).relu(), 2)
        x = F.max_pool2d(self.conv2(x).relu(), 2)
        features = self.embed(x.flatten(1)).relu()
        return self.head(features)


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
    weights = {key: tensor.numpy() for key, tensor in model.state_dict().items()}
    np.savez_compressed(output, **weights, labels=list("0123456789"))
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", default=DIGIT_MODEL, type=Path)
    args = parser.parse_args()
    train(args.archive, args.output)

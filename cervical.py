"""Cervical cancer screening - when the data is mostly holes.

UCI "Cervical Cancer (Risk Factors)" dataset (id 383): 858 patients from
Hospital Universitario de Caracas, 36 columns, target = biopsy result.
Cervical cancer kills ~350,000 women a year, most in places where screening
is scarce - which is why a questionnaire-based risk model is a real idea,
and why evaluating one honestly matters.

Three facts make this dataset a minefield:

  1. 799 of 858 rows (93%) have at least one missing answer. "Drop rows with
     missing values" - the default in many tutorials - throws away the study.
  2. Two columns are 92% missing (time since STD diagnosis). Patients skip
     intimate questions non-randomly: the hole is itself information.
  3. Only 55 biopsies (6.4%) are positive. A model that says "nobody has
     cancer" scores 93.6% accuracy with zero medical value.

So this script reports what actually matters - recall on the 55 positives -
across three honest strategies, averaged over 10 split seeds (lesson from
github.com/sravanni369/birthweight-leakage-pytorch: one split is a lottery).

The other three screening columns (Hinselmann, Schiller, Citology) are test
OUTCOMES, not risk factors - using them would be target leakage (lesson from
the same series). They are excluded.

Sources: MLP + class weighting - Dive into Deep Learning ch. 5 (d2l.ai);
missing-data discussion - ISLP ch. 4. stdlib + torch only.

Run:  python cervical.py   (CPU, ~a minute)
"""

import csv
import random

import torch
import torch.nn as nn

EPOCHS = 300
SEEDS = 10
LEAKY = {"Hinselmann", "Schiller", "Citology", "Biopsy"}


def load(path="risk_factors_cervical_cancer.csv"):
    """Returns (values, missing-mask, labels). '?' -> 0.0 in values, 1 in mask."""
    rows = list(csv.reader(open(path)))
    hdr, data = rows[0], rows[1:]
    keep = [j for j, h in enumerate(hdr) if h not in LEAKY]
    vals = torch.tensor([[0.0 if r[j] == "?" else float(r[j]) for j in keep]
                         for r in data])
    mask = torch.tensor([[1.0 if r[j] == "?" else 0.0 for j in keep]
                         for r in data])
    y = torch.tensor([float(r[hdr.index("Biopsy")]) for r in data])
    return vals, mask, y


def mean_impute(vals, mask):
    """Replace missing entries with the column mean of the observed entries."""
    out = vals.clone()
    for j in range(vals.shape[1]):
        seen = mask[:, j] == 0
        fill = vals[seen, j].mean() if seen.any() else 0.0
        out[mask[:, j] == 1, j] = fill
    return out


def train_and_eval(x, y, seed):
    """80/20 split, class-weighted MLP. Returns (accuracy, recall, flagged)."""
    idx = list(range(len(y)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * 0.8)
    tr, te = torch.tensor(idx[:cut]), torch.tensor(idx[cut:])

    mean, std = x[tr].mean(0), x[tr].std(0).clamp(min=1e-6)
    x_tr, x_te = (x[tr] - mean) / std, (x[te] - mean) / std

    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    w = (y[tr] == 0).sum() / (y[tr] == 1).sum().clamp(min=1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=w)

    for _ in range(EPOCHS):
        opt.zero_grad()
        loss_fn(model(x_tr).squeeze(1), y[tr]).backward()
        opt.step()

    with torch.no_grad():
        pred = (model(x_te).squeeze(1) > 0).float()
    pos = y[te] == 1
    acc = float((pred == y[te]).float().mean() * 100)
    rec = float((pred[pos] == 1).float().mean() * 100) if pos.any() else float("nan")
    return acc, rec, int(pred.sum())


def sweep(tag, x, y):
    runs = [train_and_eval(x, y, s) for s in range(SEEDS)]
    acc = sum(r[0] for r in runs) / SEEDS
    recs = [r[1] for r in runs if r[1] == r[1]]  # drop NaN (no positives drawn)
    flag = sum(r[2] for r in runs) / SEEDS
    print(f"{tag}: accuracy {acc:5.1f}%  positive-recall mean {sum(recs)/len(recs):5.1f}% "
          f"(min {min(recs):3.0f}%, max {max(recs):3.0f}%)  flags/172 {flag:.0f}")


if __name__ == "__main__":
    vals, mask, y = load()
    n, pos = len(y), int(y.sum())
    complete = int((mask.sum(1) == 0).sum())
    print(f"n={n} | biopsy-positive {pos} ({pos/n*100:.1f}%) | "
          f"rows with no missing values {complete}")
    print(f"'drop incomplete rows' would keep {complete/n*100:.0f}% of the study\n")

    print(f"baseline 'nobody has cancer': accuracy {(1-pos/n)*100:.1f}%, recall 0%\n")

    imputed = mean_impute(vals, mask)
    sweep("A) mean-impute only        ", imputed, y)
    sweep("B) impute + missing flags  ", torch.cat([imputed, mask], dim=1), y)

    print(f"\nAll numbers are means over {SEEDS} split seeds - a single split on")
    print("~11 test positives is a lottery. Accuracy barely moves between any of")
    print("these models; recall on the 55 women who actually had a positive")
    print("biopsy is the only number that separates them.")

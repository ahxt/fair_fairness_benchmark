"""Run a partial FFB ResNet-18 or ResNet-20 ERM/DiffDP experiment on CelebA.

The expected input is datasets/celeba/raw/celeba.csv from datasets/readme.md.
By default, a seeded reservoir sample of 20,000 rows keeps the first run modest.
Use --max-samples 0 to use every row when sufficient memory is available.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "datasets" / "celeba" / "raw" / "celeba.csv"


def preflight(csv_path: Path) -> dict:
    return {
        "data_path": str(csv_path),
        "data_present": csv_path.is_file(),
        "dependencies": {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch", "torchvision", "numpy", "pandas", "sklearn", "scipy", "statsmodels")
        },
    }


def sample_rows(csv_path: Path, max_samples: int, seed: int):
    rng = random.Random(seed)
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"pixels", "Attractive", "Male", "Smiling", "Wavy_Hair", "Young"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CelebA columns: {sorted(missing)}")
        for seen, row in enumerate(reader):
            item = tuple(row[name] for name in ("pixels", "Smiling", "Wavy_Hair", "Attractive", "Male", "Young"))
            if not max_samples or len(rows) < max_samples:
                rows.append(item)
            else:
                idx = rng.randrange(seen + 1)
                if idx < max_samples:
                    rows[idx] = item
    return rows, seen + 1


def main(default_architecture: str = "resnet18", default_method: str = "erm") -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--architecture", choices=("resnet18", "resnet20"), default=default_architecture)
    parser.add_argument("--method", choices=("erm", "diffdp"), default=default_method)
    parser.add_argument("--lam", type=float, default=1.0, help="DiffDP fairness weight; repository default is 1.0")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weights", choices=("imagenet", "random"))
    parser.add_argument("--check", action="store_true", help="Check readiness without training")
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be nonnegative")
    if args.lam < 0:
        parser.error("--lam must be nonnegative")
    if args.weights is None:
        args.weights = "imagenet" if args.architecture == "resnet18" else "random"
    if args.architecture == "resnet20" and args.weights != "random":
        parser.error("resnet_20.py has no ImageNet weights; use --weights random")
    if args.output is None:
        suffix = "" if args.method == "erm" else f"_diffdp_lam{args.lam:g}"
        args.output = ROOT / "results" / f"{args.architecture}_celeba_attractive_gender{suffix}.json"

    readiness = preflight(args.data)
    if args.check:
        print(json.dumps(readiness, indent=2))
        return 0 if readiness["data_present"] and all(readiness["dependencies"].values()) else 1
    if not readiness["data_present"] or not all(readiness["dependencies"].values()):
        print(json.dumps(readiness, indent=2), file=sys.stderr)
        print("Training requires the dataset and all listed dependencies.", file=sys.stderr)
        return 2

    os.environ.setdefault("TORCH_HOME", str(ROOT / "weights"))
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    sys.path.insert(0, str(ROOT / "src"))
    from metrics import metric_evaluation
    from loss import DiffDP

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rows, total_rows = sample_rows(args.data, args.max_samples, args.seed)
    # Match src/dataset.py: 3 x 48 x 48 pixels divided by 255.
    pixels = np.stack([np.fromstring(row[0], sep=" ", dtype=np.float32) for row in rows])
    if pixels.shape[1] != 3 * 48 * 48:
        raise ValueError(f"Expected 6912 pixel values per row; got {pixels.shape[1]}")
    images = (pixels / 255.0).reshape(-1, 3, 48, 48)
    attr = np.asarray([[float(v) for v in row[1:]] for row in rows], dtype=np.float32)
    indices = np.arange(len(rows))
    stratify = attr if args.max_samples == 0 else attr[:, [2, 3]]
    train_idx, hold_idx = train_test_split(indices, test_size=0.2, stratify=stratify, random_state=args.seed)
    test_idx, val_idx = train_test_split(
        hold_idx, test_size=0.5, stratify=stratify[hold_idx], random_state=args.seed
    )

    def loader(idx, batch_size, shuffle=False):
        return DataLoader(
            TensorDataset(torch.from_numpy(images[idx]), torch.from_numpy(attr[idx])),
            batch_size=batch_size, shuffle=shuffle, num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = loader(train_idx, args.batch_size, shuffle=True)
    val_loader = loader(val_idx, args.eval_batch_size)
    test_loader = loader(test_idx, args.eval_batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.architecture == "resnet18":
        from networks import resnet18_encoder
        model = resnet18_encoder(pretrained=args.weights == "imagenet")
    else:
        from resnet_20 import resnet20
        backbone = resnet20()
        backbone.linear = torch.nn.Linear(64, 1)
        torch.nn.init.kaiming_normal_(backbone.linear.weight)
        model = torch.nn.Sequential(backbone, torch.nn.Sigmoid())
    model = model.to(device)

    def predict(x):
        output = model(x)
        return output[1] if isinstance(output, tuple) else output

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    criterion = torch.nn.BCELoss()
    fair_criterion = DiffDP() if args.method == "diffdp" else None
    batches = iter(train_loader)
    last_loss = None
    last_clf_loss = None
    last_fair_loss = None
    for step in range(args.steps):
        try:
            x, a = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            x, a = next(batches)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = predict(x.to(device))
        clf_loss = criterion(pred, a[:, 2:3].to(device))
        if fair_criterion is None:
            fair_loss = None
            loss = clf_loss
        else:
            fair_loss = fair_criterion(pred, a[:, 3:4].to(device))
            if not torch.isfinite(fair_loss):
                raise ValueError("DiffDP loss is non-finite; check both sensitive groups in the batch")
            loss = clf_loss + args.lam * fair_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.item())
        last_clf_loss = float(clf_loss.item())
        last_fair_loss = None if fair_loss is None else float(fair_loss.item())
        if (step + 1) % 25 == 0 or step + 1 == args.steps:
            print(f"step={step+1} loss={last_loss:.5f} clf={last_clf_loss:.5f} "
                  f"fair={last_fair_loss} lr={scheduler.get_last_lr()[0]:.6g}", flush=True)

    def evaluate(data_loader, prefix):
        probs, targets, groups = [], [], []
        model.eval()
        with torch.no_grad():
            for x, a in data_loader:
                p = predict(x.to(device))
                probs.append(p.cpu().numpy())
                targets.append(a[:, 2:3].numpy())
                groups.append(a[:, 3:4].numpy())
        return metric_evaluation(np.concatenate(targets), np.concatenate(probs),
                                 np.concatenate(groups), prefix=prefix)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_suffix(".pt")
    torch.save(model.state_dict(), checkpoint)
    result = {
        "status": "completed", "model": f"{args.architecture} {args.method.upper()}",
        "method": args.method, "fairness_weight": args.lam if args.method == "diffdp" else None,
        "dataset": "CelebA-A",
        "architecture_source": "resnet_20.py" if args.architecture == "resnet20" else "src/networks.py",
        "target": "Attractive", "sensitive_attribute": "Male", "seed": args.seed,
        "total_csv_rows": total_rows, "sampled_rows": len(rows),
        "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.lr,
        "weights": args.weights, "device": str(device),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "checkpoint": str(checkpoint),
        "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "last_train_loss": last_loss, "last_classifier_loss": last_clf_loss,
        "last_fairness_loss": last_fair_loss, "validation": evaluate(val_loader, "val"),
        "test": evaluate(test_loader, "test"),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "test": result["test"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

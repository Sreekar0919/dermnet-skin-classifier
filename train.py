"""Train and evaluate an ImageNet-pretrained classifier on DermNet (23 classes).

Runs on Apple-silicon GPU (PyTorch MPS), falling back to CUDA or CPU.
Phase 1 trains only a new head on a frozen ImageNet backbone; phase 2 fine-tunes
the whole network. A resumable checkpoint is written after every epoch
(``--resume`` continues from it). The checkpoint with the best validation macro F1 (validation
is a stratified 10% split of train) is evaluated once on the held-out test set.

    python train.py --arch efficientnet_b0   # or convnext_tiny, resnet50
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet stats the backbone expects


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(data_dir, img_size, batch_size, workers, seed):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_aug = datasets.ImageFolder(Path(data_dir) / "train", train_tf)
    train_plain = datasets.ImageFolder(Path(data_dir) / "train", eval_tf)
    test = datasets.ImageFolder(Path(data_dir) / "test", eval_tf)
    if train_aug.classes != test.classes:
        raise ValueError("train/ and test/ class folders differ")

    idx = np.arange(len(train_aug))
    tr_idx, va_idx = train_test_split(idx, test_size=0.1, stratify=train_aug.targets, random_state=seed)
    kw = dict(batch_size=batch_size, num_workers=workers, persistent_workers=workers > 0)
    return (
        DataLoader(Subset(train_aug, tr_idx), shuffle=True, drop_last=True, **kw),
        DataLoader(Subset(train_plain, va_idx), **kw),
        DataLoader(test, **kw),
        train_aug.classes,
    )


ARCHS = {
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


def head_of(model):
    return model.fc if hasattr(model, "fc") else model.classifier


def build_model(arch, num_classes):
    fn, weights = ARCHS[arch]
    model = fn(weights=weights)
    new_head = lambda in_f: nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    if arch == "resnet50":
        model.fc = new_head(model.fc.in_features)
    elif arch == "efficientnet_b0":
        model.classifier = new_head(model.classifier[1].in_features)
    else:  # convnext keeps its LayerNorm + Flatten, only the final Linear is replaced
        model.classifier[2] = new_head(model.classifier[2].in_features)
    return model


def set_backbone_trainable(model, trainable):
    for p in model.parameters():
        p.requires_grad = trainable
    for p in head_of(model).parameters():
        p.requires_grad = True


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def run_phase(name, model, loaders, device, epochs, lr, history, best, resume, last_ckpt):
    train_loader, val_loader = loaders
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_loader), pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    done = sum(h["phase"] == name for h in history)
    if done and resume and resume["phase"] == name:
        opt.load_state_dict(resume["opt"])
        sched.load_state_dict(resume["sched"])
    for epoch in range(done, epochs):
        model.train()
        t0, total, correct, loss_sum = time.time(), 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        logits, labels = predict(model, val_loader, device)
        preds = logits.argmax(1)
        row = {
            "phase": name,
            "epoch": len(history) + 1,
            "train_loss": loss_sum / total,
            "train_acc": correct / total,
            "val_loss": nn.functional.cross_entropy(logits, labels).item(),
            "val_acc": (preds == labels).float().mean().item(),
            "val_macro_f1": f1_score(labels, preds, average="macro"),
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_macro_f1"] > best["f1"]:
            best.update(f1=row["val_macro_f1"], epoch=row["epoch"])
            torch.save(model.state_dict(), best["path"])
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "phase": name, "history": history, "best": best}, last_ckpt)


def plot_history(history, out):
    ep = [h["epoch"] for h in history]
    fine_start = next((h["epoch"] for h in history if h["phase"] == "finetune"), None)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "loss", "Loss"), (axes[1], "acc", "Top-1 accuracy")):
        ax.plot(ep, [h[f"train_{key}"] for h in history], label="train")
        ax.plot(ep, [h[f"val_{key}"] for h in history], label="validation")
        if fine_start:
            ax.axvline(fine_start - 0.5, color="gray", ls="--", lw=1, label="unfreeze backbone")
        ax.set(title=title, xlabel="epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_confusion(cm, classes, out):
    norm = cm / cm.sum(axis=1, keepdims=True)
    short = [c.split(" ")[0][:18] for c in classes]
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), short, rotation=90, fontsize=8)
    ax.set_yticks(range(len(classes)), short, fontsize=8)
    ax.set(xlabel="predicted", ylabel="true", title="Test confusion matrix (row-normalised)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def evaluate(model, test_loader, classes, device, out_dir):
    logits, labels = predict(model, test_loader, device)
    preds = logits.argmax(1)
    top5 = logits.topk(5, dim=1).indices
    metrics = {
        "test_images": len(labels),
        "num_classes": len(classes),
        "top1_accuracy": (preds == labels).float().mean().item(),
        "top5_accuracy": (top5 == labels[:, None]).any(1).float().mean().item(),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }
    (out_dir / "classification_report.txt").write_text(
        classification_report(labels, preds, target_names=classes, digits=3)
    )
    cm = confusion_matrix(labels, preds, labels=range(len(classes)))
    (out_dir / "confusion_matrix.json").write_text(json.dumps({"classes": classes, "matrix": cm.tolist()}))
    plot_confusion(cm, classes, out_dir / "confusion_matrix.png")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dermnet")
    ap.add_argument("--arch", choices=sorted(ARCHS), default="efficientnet_b0")
    ap.add_argument("--out", help="results folder (default: results/<arch>)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--finetune-lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/<arch>/last.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out or f"results/{args.arch}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints") / args.arch
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.pt"
    train_loader, val_loader, test_loader, classes = make_loaders(
        args.data, args.img_size, args.batch_size, args.workers, args.seed
    )
    print(f"device={device} classes={len(classes)} train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}", flush=True)

    model = build_model(args.arch, len(classes)).to(device)
    history, best = [], {"f1": -1.0, "epoch": None, "path": ckpt_dir / "best.pt"}
    resume = torch.load(last_ckpt, map_location=device, weights_only=False) if args.resume else None
    if resume:
        model.load_state_dict(resume["model"])
        history, best = resume["history"], resume["best"]
        print(f"resumed after epoch {len(history)}", flush=True)
    loaders = (train_loader, val_loader)

    set_backbone_trainable(model, False)
    run_phase("head", model, loaders, device, args.head_epochs, args.head_lr, history, best, resume, last_ckpt)
    set_backbone_trainable(model, True)
    run_phase("finetune", model, loaders, device, args.finetune_epochs, args.finetune_lr, history, best, resume, last_ckpt)

    model.load_state_dict(torch.load(best["path"], map_location=device))
    metrics = evaluate(model, test_loader, classes, device, out_dir)
    metrics.update(best_epoch=best["epoch"], config=vars(args), device=str(device),
                   train_minutes=round(sum(h["seconds"] for h in history) / 60, 1))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, out_dir / "training_curves.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

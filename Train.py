"""
ENGE707 Assessment 3 - cat/dog facial feature classifier (8 classes).

Pipeline: stratified train/val split -> pretrained timm CNN fine-tuned end to end
-> learning curves + validation metrics -> Kaggle submission.csv.

Run from this folder with the .venv active:
    python train.py --limit 2000 --epochs 2      # quick smoke test (skips test predictions)
    python train.py                              # full run with defaults
    python train.py --model resnet18 --img-size 128 --run-name r18_128   # an alternative experiment
    python train.py --exclude too_small,blurry --group-split            # Task 2 options (need outputs/quality_flags.csv)

Each run writes to runs/<run-name>/:
    config.json           every setting used (reproducibility)
    split.csv             which images went to train / val
    history.csv           per-epoch train/val loss + accuracy (learning curves)
    best.pt               weights from the epoch with the lowest validation loss
    val_report.txt        precision / recall / F1 per class
    confusion_matrix.csv  rows = true class, columns = predicted class
    val_predictions.csv   every validation image with its prediction (error analysis)
    submission.csv        Kaggle file: image_id,class
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / 'enge-707-2026-s-2-cat-and-dog-facial-feature-classification-ch'
TRAIN_DIR = DATA_DIR / 'train' / 'train'
TEST_DIR = DATA_DIR / 'test' / 'test'


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='efficientnet_b0', help='any timm model name')
    p.add_argument('--img-size', type=int, default=160)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4, help='peak learning rate')
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--val-frac', type=float, default=0.15)
    p.add_argument('--patience', type=int, default=3, help='early stopping: epochs without val-loss improvement')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-amp', action='store_true', help='disable bfloat16 mixed precision')
    p.add_argument('--no-pretrained', action='store_true', help='train from random weights (comparison baseline)')
    p.add_argument('--limit', type=int, default=0, help='use only N training images (smoke test)')
    p.add_argument('--exclude', default='', help='quality flags to drop, e.g. too_small,blurry')
    p.add_argument('--group-split', action='store_true', help='keep duplicate images (same phash) on one side of the split')
    p.add_argument('--run-name', default=None)
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device():
    # Intel Arc on the laptop, NVIDIA on Kaggle, CPU as a last resort
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class ImageDataset(Dataset):
    """Loads <folder>/<id>.jpg on demand. targets=None for the unlabelled test set."""

    def __init__(self, folder, ids, targets, tfm):
        self.folder, self.ids, self.targets, self.tfm = folder, list(ids), targets, tfm

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self.folder / f'{self.ids[i]}.jpg').convert('RGB')  # some JPEGs may be greyscale
        y = -1 if self.targets is None else int(self.targets[i])
        return self.tfm(img), y


def build_transforms(size, mean, std):
    # Augmentation only on training images; val/test get the same deterministic resize + normalise.
    # Horizontal flip is label-safe (a left eye is still an eye). Crops are kept mild (>=70% of the
    # image) so the feature that defines the class isn't cropped away.
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean, std),  # ImageNet statistics, matching the pretrained weights
    ])
    evaluate = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train, evaluate


def run_epoch(model, loader, device, criterion, amp, optimizer=None, scheduler=None):
    """One pass over loader. Trains if an optimizer is given, otherwise just evaluates."""
    training = optimizer is not None
    model.train(training)  # toggles dropout / batch-norm behaviour
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False, desc='train' if training else 'val'):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()  # OneCycle schedule steps every batch
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device, amp):
    """Returns class probabilities, shape (n_images, n_classes), in loader order."""
    model.eval()
    probs = []
    for x, _ in tqdm(loader, leave=False, desc='predict'):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x.to(device))
        probs.append(logits.float().softmax(1).cpu())
    return torch.cat(probs).numpy()


def main():
    a = get_args()
    seed_everything(a.seed)
    device = pick_device()
    amp = not a.no_amp and device.type != 'cpu'
    run = a.run_name or f'{a.model}_{a.img_size}_{time.strftime("%m%d-%H%M")}'
    out = ROOT / 'runs' / run
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(a), indent=2))
    print(f'device={device}  amp={amp}  output={out}')

    # ---- labels + split -------------------------------------------------
    labels = pd.read_csv(DATA_DIR / 'train_labels.csv', dtype={'image_id': str})  # keep leading zeros
    classes = sorted(labels['class'].unique())           # official spellings, taken from the data
    labels['y'] = labels['class'].map({c: i for i, c in enumerate(classes)})   # label encoding: class -> 0..7

    # Optional Task 2 cleaning: flags come from the Task 1 notebook (cell 5h)
    if a.exclude or a.group_split:
        flags = pd.read_csv(ROOT / 'outputs' / 'quality_flags.csv', dtype={'image_id': str})
        labels = labels.merge(flags.drop(columns='class'), on='image_id')
    if a.exclude:
        drop = labels[a.exclude.split(',')].any(axis=1)    # True if the image has any of the listed flags
        print(f'excluding {drop.sum()} flagged training images')
        labels = labels[~drop]                             # test images are never removed

    if a.limit:
        labels = labels.sample(a.limit, random_state=a.seed)

    if a.group_split:
        # Visual duplicates share a phash: keep each group on one side so none leak into validation
        folds = StratifiedGroupKFold(n_splits=round(1 / a.val_frac), shuffle=True, random_state=a.seed)
        tr_idx, va_idx = next(folds.split(labels, labels['y'], groups=labels['phash']))
        tr, va = labels.iloc[tr_idx], labels.iloc[va_idx]
    else:
        # Stratified: every class keeps the same proportion in train and val
        tr, va = train_test_split(labels, test_size=a.val_frac, stratify=labels['y'], random_state=a.seed)
    print(f'train={len(tr)}  val={len(va)}  classes={classes}')
    pd.concat([tr.assign(set='train'), va.assign(set='val')])[['image_id', 'class', 'set']] \
      .to_csv(out / 'split.csv', index=False)

    # ---- model ------------------------------------------------------------
    # Transfer learning: ImageNet-pretrained backbone, new 8-way classification head.
    model = timm.create_model(a.model, pretrained=not a.no_pretrained, num_classes=len(classes)).to(device)
    cfg = timm.data.resolve_model_data_config(model)
    train_tfm, eval_tfm = build_transforms(a.img_size, cfg['mean'], cfg['std'])
    print(f'{a.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters')

    dl_kw = dict(batch_size=a.batch_size, num_workers=a.workers, persistent_workers=a.workers > 0)
    train_dl = DataLoader(ImageDataset(TRAIN_DIR, tr['image_id'], tr['y'].values, train_tfm),
                          shuffle=True, drop_last=True, **dl_kw)
    val_dl = DataLoader(ImageDataset(TRAIN_DIR, va['image_id'], va['y'].values, eval_tfm),
                        shuffle=False, **dl_kw)

    # Label smoothing + weight decay regularise against overfitting. The same criterion is used
    # for train and val so the two loss curves are directly comparable.
    criterion = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    # OneCycle: short warm-up (10% of steps) then cosine decay - protects pretrained weights early on.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=a.lr, total_steps=a.epochs * len(train_dl), pct_start=0.1)

    # ---- training loop with early stopping ----------------------------------
    history, best_loss, bad_epochs = [], float('inf'), 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, device, criterion, amp, optimizer, scheduler)
        va_loss, va_acc = run_epoch(model, val_dl, device, criterion, amp)
        secs = time.time() - t0
        history.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, lr=optimizer.param_groups[0]['lr'], secs=secs))
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)  # saved every epoch
        print(f'epoch {ep:2d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | '
              f'val loss {va_loss:.4f} acc {va_acc:.4f} | {secs:.0f}s')

        if va_loss < best_loss:
            best_loss, bad_epochs = va_loss, 0
            torch.save(model.state_dict(), out / 'best.pt')
        else:
            bad_epochs += 1
            if bad_epochs >= a.patience:
                print(f'early stopping: no val-loss improvement for {a.patience} epochs')
                break

    # ---- validation evaluation with the best checkpoint ---------------------
    model.load_state_dict(torch.load(out / 'best.pt', map_location=device))
    va_prob = predict(model, val_dl, device, amp)
    va_pred = va_prob.argmax(1)
    report = classification_report(va['y'], va_pred, labels=range(len(classes)),
                                   target_names=classes, digits=4, zero_division=0)
    print(report)
    (out / 'val_report.txt').write_text(report)
    pd.DataFrame(confusion_matrix(va['y'], va_pred, labels=range(len(classes))),
                 index=classes, columns=classes).to_csv(out / 'confusion_matrix.csv')
    va.assign(pred=[classes[i] for i in va_pred], confidence=va_prob.max(1)) \
      .to_csv(out / 'val_predictions.csv', index=False)

    if a.limit:
        print('smoke test (--limit) - skipping test predictions')
        return

    # ---- Kaggle test predictions ---------------------------------------------
    test_ids = sorted(p.stem for p in TEST_DIR.glob('*.jpg'))
    test_dl = DataLoader(ImageDataset(TEST_DIR, test_ids, None, eval_tfm), shuffle=False, **dl_kw)
    te_pred = predict(model, test_dl, device, amp).argmax(1)
    preds = pd.DataFrame({'image_id': [int(i) for i in test_ids], 'class': [classes[i] for i in te_pred]})

    # Match sample_submission.csv exactly: same IDs (unpadded ints), same row order.
    sample = pd.read_csv(DATA_DIR / 'sample_submission.csv')
    sub = sample[['image_id']].merge(preds, on='image_id', how='left')
    assert sub['class'].notna().all(), 'some test IDs have no prediction'
    sub.to_csv(out / 'submission.csv', index=False)
    print(f'wrote {out / "submission.csv"} ({len(sub)} rows)')
    print(sub['class'].value_counts())


if __name__ == '__main__':  # required on Windows when DataLoader uses worker processes
    main()
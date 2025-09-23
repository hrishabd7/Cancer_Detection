# train_torch.py
# ---------------------------------------------------------
# CT-scan classifier on folder layout:
#   data/Data/{train,valid,test}/<class>/*.{png,jpg}
# Upgrades: ConvNeXt-Tiny @512, CT-safe augments, no sampler,
# label smoothing (annealed), AdamW + warmup+cosine, SWA tail,
# tiny TTA, "uncertain" flag, CUDA/MPS-safe loaders,
# optional Focal Loss, test-time ensemble (best+last+SWA).
# ---------------------------------------------------------
import os, math, random
from typing import Tuple, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from PIL import Image
import random
import matplotlib.pyplot as plt


# ========= Paths & Config =========
ROOT      = os.path.join(os.getcwd(), "data", "Data")
TRAIN_DIR = os.path.join(ROOT, "train")
VAL_DIR   = os.path.join(ROOT, "valid")
TEST_DIR  = os.path.join(ROOT, "test")

IMG_SIZE   = 512         # higher resolution for subtle lung patterns
BATCH      = 10          # adjust down if OOM on MPS/CPU
EPOCHS     = 15          # phase 1: head warmup
EPOCHS_FT  = 25          # phase 2: full fine-tune
LR_HEAD    = 5e-4
LR_FT      = 5e-5
WD         = 1e-4        # AdamW weight decay
SMOOTH_START = 0.02      # label smoothing at start
SMOOTH_END   = 0.0       # anneal to 0 near the end
SWA_EPOCHS   = 5         # last N FT epochs use SWA averaging
UNCERTAIN_THRESH = 0.60  # flag low-confidence predictions
USE_FOCAL   = False      # set True if cancer-vs-cancer confusion persists
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ========= Utilities =========
def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int | None, device: torch.device) -> None:
    if seed is None: return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)

# ImageNet stats (keep — ConvNeXt was trained on these)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Simple lung window mapping for tensors (approx if you don't have HU)
def window_lung_tensor(t: torch.Tensor, level: float = -600., width: float = 1500.):
    """
    t: 1xHxW in [0,1] (grayscale). We pseudo-map to HU via: hu ~= t*4096 - 1024.
    Then clamp to lung window and re-normalize 0..1.
    """
    hu = t * 4096.0 - 1024.0
    lo, hi = level - width / 2.0, level + width / 2.0
    hu = torch.clamp(hu, lo, hi)
    return (hu - lo) / max(1e-6, (hi - lo))  # back to 0..1

def make_transforms(img_size: int):
    """CT-safe training/eval transforms with gentle color jitter & erasing."""
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),                                  # 1xHxW
        transforms.Lambda(lambda t: window_lung_tensor(t)),     # stabilize contrast
        transforms.Lambda(lambda t: t.repeat(3, 1, 1)),         # 3ch for ConvNeXt
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), shear=5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.10)], p=0.3),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.08), ratio=(0.5, 2.0)),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: window_lung_tensor(t)),
        transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf

def build_model(num_classes: int) -> nn.Module:
    """ConvNeXt-Tiny backbone with new classifier head."""
    m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_feats = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_feats, num_classes)
    return m

class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup for 'warmup' epochs then cosine to epoch_end (per-epoch stepping)."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = max(warmup_epochs, 0)
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]
        t = self.last_epoch - self.warmup_epochs
        T = max(1, self.total_epochs - self.warmup_epochs)
        return [base_lr * 0.5 * (1 + math.cos(math.pi * t / T))
                for base_lr in self.base_lrs]

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        ce = F.nll_loss(logp, target, weight=self.weight, reduction='none')
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

def tta_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Average logits over tiny TTA (id, hflip, ±5°)."""
    ops: List[Callable[[torch.Tensor], torch.Tensor]] = [
        lambda z: z,
        lambda z: TF.hflip(z),
        lambda z: TF.rotate(z, angle=5),
        lambda z: TF.rotate(z, angle=-5),
    ]
    with torch.no_grad():
        out = None
        for op in ops:
            y = model(op(x))
            out = y if out is None else (out + y)
        return out / len(ops)

def get_loader_kwargs(device: torch.device):
    """CUDA: parallel workers; MPS/CPU: single-worker to avoid spawn issues."""
    if device.type == "cuda":
        cpu_half = (os.cpu_count() or 2) // 2
        num_workers = min(8, max(2, cpu_half))
        return dict(num_workers=num_workers, pin_memory=True, persistent_workers=True)
    else:
        return dict(num_workers=0, pin_memory=False, persistent_workers=False)

def run_one_epoch(model, loader, criterion, device, optimizer=None) -> Tuple[float, float]:
    train = optimizer is not None
    model.train() if train else model.eval()
    total, correct, running_loss = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for images, labels in tqdm(loader, ncols=80, leave=False):
            images, labels = images.to(device), labels.to(device)
            if train: optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct      += (outputs.argmax(1) == labels).sum().item()
            total        += labels.size(0)
    return running_loss / max(1, total), correct / max(1, total)


# ========= Main =========
def main():
    device = get_device()
    print(f"Using device: {device}")
    set_seed(SEED, device)

    # Transforms & datasets
    train_tf, eval_tf = make_transforms(IMG_SIZE)
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=eval_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=eval_tf)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    # DataLoaders (WeightedRandomSampler for class balancing in train)
    loader_kwargs = get_loader_kwargs(device)
    # Compute class weights for sampler
    targets = [label for _, label in train_ds.samples]
    class_counts = np.bincount(targets, minlength=len(class_names)).astype(np.float64)
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights[targets]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(targets),
        replacement=True
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH, sampler=sampler, shuffle=False, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH, shuffle=False, **loader_kwargs
    )

    # Model
    model = build_model(num_classes).to(device)

    # ----- Phase 1: train head only -----
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("classifier.")  # only final head

    head_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(head_params, lr=LR_HEAD, weight_decay=WD)
    sched = WarmupCosine(opt, warmup_epochs=1, total_epochs=EPOCHS)

    if USE_FOCAL:
        criterion = FocalLoss(gamma=1.5)  # no label smoothing with focal
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=SMOOTH_START)

    best_path = "best_convnext_tiny.pt"
    last_path = "last_convnext_tiny.pt"
    swa_path  = "swa_convnext_tiny.pt"
    best_val = 0.0

    print("\n=== Phase 1: head warmup ===")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, device, optimizer=opt)
        va_loss, va_acc = run_one_epoch(model, val_loader,   criterion, device, optimizer=None)
        sched.step()
        print(f"Epoch {epoch:02d}/{EPOCHS} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ saved new BEST (val_acc={best_val:.3f})")

    # ----- Phase 2: unfreeze full model -----
    print("\n=== Phase 2: full fine-tune ===")
    model.load_state_dict(torch.load(best_path, map_location=device))
    for p in model.parameters(): p.requires_grad = True

    opt = torch.optim.AdamW(model.parameters(), lr=LR_FT, weight_decay=WD)
    sched = WarmupCosine(opt, warmup_epochs=2, total_epochs=EPOCHS_FT)
    swa_model = AveragedModel(model) if SWA_EPOCHS > 0 else None
    swa_sched = SWALR(opt, swa_lr=LR_FT/10) if SWA_EPOCHS > 0 else None

    for epoch in range(1, EPOCHS_FT + 1):
        # anneal smoothing to 0 in the last 5 epochs unless using focal
        if not USE_FOCAL:
            smooth = SMOOTH_START if epoch <= max(0, EPOCHS_FT - 5) else SMOOTH_END
            criterion = nn.CrossEntropyLoss(label_smoothing=smooth)
        else:
            criterion = FocalLoss(gamma=1.5)

        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, device, optimizer=opt)
        va_loss, va_acc = run_one_epoch(model, val_loader,   criterion, device, optimizer=None)
        sched.step()

        # SWA tail
        if SWA_EPOCHS > 0 and epoch > (EPOCHS_FT - SWA_EPOCHS):
            swa_model.update_parameters(model)
            swa_sched.step()

        print(f"[FT] {epoch:02d}/{EPOCHS_FT} | train {tr_loss:.4f}/{tr_acc:.3f} | val {va_loss:.4f}/{va_acc:.3f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ saved new BEST (val_acc={best_val:.3f})")

    # Save last model too
    torch.save(model.state_dict(), last_path)

    # Finalize SWA (if used)
    swa_ready = False
    if swa_model is not None:
        print("Updating BN stats for SWA model...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), swa_path)
        swa_ready = True

    # ----- Test with tiny TTA + (best + last + swa) ensemble -----
    print("\n=== Test (tiny TTA + ensemble) ===")
    logits_collect = []

    # best
    m_best = build_model(num_classes).to(device)
    m_best.load_state_dict(torch.load(best_path, map_location=device))
    m_best.eval()

    # last
    m_last = build_model(num_classes).to(device)
    m_last.load_state_dict(torch.load(last_path, map_location=device))
    m_last.eval()

    # swa (optional)
    models_for_test = [m_best, m_last]
    if swa_ready:
        def strip_module_prefix(state_dict):
            """Removes 'module.' prefix from keys in a state dict if present."""
            return {k.replace('module.', '') if k.startswith('module.') else k: v for k, v in state_dict.items()}
        m_swa = build_model(num_classes).to(device)
        swa_state = torch.load(swa_path, map_location=device)
        swa_state.pop("n_averaged", None)  # Remove SWA-specific key if present
        swa_state = strip_module_prefix(swa_state)
        m_swa.load_state_dict(swa_state)
        m_swa.eval()
        models_for_test.append(m_swa)

    y_true, y_pred = [], []
    n_uncertain = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, ncols=80, leave=False):
            images = images.to(device)

            # average logits across (models × TTA views)
            logits_ens = None
            for m in models_for_test:
                l = tta_logits(m, images)
                logits_ens = l if logits_ens is None else (logits_ens + l)
            logits_ens /= float(len(models_for_test))

            probs = F.softmax(logits_ens, dim=1)
            confs, preds = probs.max(dim=1)
            n_uncertain += (confs < UNCERTAIN_THRESH).sum().item()

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(f"Uncertain (< {UNCERTAIN_THRESH:.2f}) predictions: {n_uncertain} / {len(test_ds)}")

    # === Interactive prediction ===
    interactive_predict(model, test_ds, device, class_names, eval_tf)

def interactive_predict(model, dataset, device, class_names, eval_tf):
    """Pop up random images and show model prediction."""
    model.eval()
    while True:
        idx = random.randint(0, len(dataset) - 1)
        img_path, true_label = dataset.samples[idx]
        img = Image.open(img_path)
        img_tensor = eval_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = logits.argmax(1).item()
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_idx]} ({probs[pred_idx]:.2f})")
        plt.axis('off')
        plt.show()
        cont = input("Show another image? (y/n): ")
        if cont.lower() != 'y':
            break






# ========= Main guard =========
if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()



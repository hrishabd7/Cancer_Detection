# train_torch.py
# ---------------------------------------------------------
# CT-scan classifier on folder layout:
#   data/Data/{train,valid,test}/<class>/*.{png,jpg}
# Upgrades: ConvNeXt-Tiny @512, CT-safe augments, no sampler,
# label smoothing (annealed), AdamW + warmup+cosine, SWA tail,
# tiny TTA, "uncertain" flag, CUDA/MPS-safe loaders,
# optional Focal Loss, test-time ensemble (best+last+SWA).
# ---------------------------------------------------------
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from utils import get_device, set_seed, window_lung_tensor, make_transforms, get_loader_kwargs
from modeling import build_model, WarmupCosine, FocalLoss
from training import run_one_epoch, tta_logits
from interactive import interactive_predict
from dataloader import get_datasets_and_loaders
from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMG_SIZE, BATCH, EPOCHS, EPOCHS_FT, LR_HEAD, LR_FT, WD,
    SMOOTH_START, SMOOTH_END, SWA_EPOCHS, UNCERTAIN_THRESH, USE_FOCAL, SEED,
    best_path, last_path, swa_path
)


# ========= Main =========
def main():
    device = get_device()
    print(f"Using device: {device}")
    set_seed(SEED, device)

    # Transforms & datasets
    train_tf, eval_tf = make_transforms(IMG_SIZE)
    loader_kwargs = get_loader_kwargs(device)

    # Create datasets first
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=eval_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=eval_tf)

    class_names = train_ds.classes

    # Now call get_datasets_and_loaders if needed
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = get_datasets_and_loaders(
        TRAIN_DIR, VAL_DIR, TEST_DIR, train_tf, eval_tf, BATCH, loader_kwargs, class_names
    )

    num_classes = len(class_names)
    print("Classes:", class_names)

    # instantiate model now that num_classes is known
    # Use default head (matches saved checkpoints). Change to feature_dim only if you retrain.
    model = build_model(num_classes, feature_dim=512).to(device)   # custom 512 head
    # or keep default backbone head (768):
    # model = build_model(num_classes).to(device)
    #

    # DataLoaders (WeightedRandomSampler for class balancing in train)
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

    # --- Extract, pad and save penultimate features (optional) ---
    import joblib
    NPY_DIR = os.path.join(os.getcwd(), "npy_outputs")
    os.makedirs(NPY_DIR, exist_ok=True)

    def pad_or_truncate(arr: np.ndarray, target_dim: int):
        if arr.size == 0:
            return np.zeros((0, target_dim), dtype=np.float32)
        D = arr.shape[1]
        if D == target_dim:
            return arr
        if D < target_dim:
            pad = np.zeros((arr.shape[0], target_dim - D), dtype=arr.dtype)
            return np.concatenate([arr, pad], axis=1)
        return arr[:, :target_dim]

    def extract_and_save_padded_features(model, splits, transform, batch_size, loader_kwargs, target_dim=512):
        backbone = getattr(model, "backbone", model)
        if not hasattr(backbone, "features") or not hasattr(backbone, "avgpool"):
            print("Backbone does not expose features/avgpool; skipping feature extraction.")
            return
        feat_extractor = torch.nn.Sequential(backbone.features, backbone.avgpool, torch.nn.Flatten()).to(device)
        feat_extractor.eval()

        for name, split_dir in splits.items():
            ds = datasets.ImageFolder(split_dir, transform=transform)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
            feats = []
            with torch.no_grad():
                for imgs, _ in dl:
                    imgs = imgs.to(device)
                    f = feat_extractor(imgs)
                    feats.append(f.cpu().numpy())
            feats = np.concatenate(feats, axis=0) if len(feats) else np.zeros((0, 0), dtype=np.float32)
            feats_p = pad_or_truncate(feats, target_dim)
            out_name = os.path.join(NPY_DIR, f"features_{target_dim}_{name}.npy")
            np.save(out_name, feats_p)
            print(f"Saved padded features: {out_name} (shape={feats_p.shape})")

    # Call the extractor if you want padded features saved.
    # Set target_dim to 1024 (or 512) as needed.
    extract_and_save_padded_features(model, {
        "train": TRAIN_DIR,
        "valid": VAL_DIR,
        "test": TEST_DIR,
    }, eval_tf, BATCH, loader_kwargs, target_dim=512)

    # ----- Test with tiny TTA + (best + last + swa) ensemble -----
    print("\n=== Test (tiny TTA + ensemble) ===")
    logits_collect = []

    # best
    m_best = build_model(num_classes, feature_dim=512).to(device)   
    m_best.load_state_dict(torch.load(best_path, map_location=device))
    m_best.eval()

    # last
    m_last = build_model(num_classes, feature_dim=512).to(device)   
    m_last.load_state_dict(torch.load(last_path, map_location=device))
    m_last.eval()

    # swa (optional)
    models_for_test = [m_best, m_last]
    if swa_ready:
        def strip_module_prefix(state_dict):
            """Removes 'module.' prefix from keys in a state dict if present."""
            return {k.replace('module.', '') if k.startswith('module.') else k: v for k, v in state_dict.items()}
        m_swa = build_model(num_classes, feature_dim=512).to(device)   
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



# ========= Main guard =========
if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

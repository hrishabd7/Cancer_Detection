import torch
import os
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader

from config import IMG_SIZE, TRAIN_DIR, VAL_DIR, TEST_DIR, best_path
from utils import get_device, make_transforms
from modeling import build_model

from sklearn.decomposition import PCA
import joblib

# Output file templates
NPY_DIR = os.path.join(os.getcwd(), "npy_outputs")
os.makedirs(NPY_DIR, exist_ok=True)

def out_path(name, split, ext):
    return os.path.join(NPY_DIR, f"{name}_{split}.{ext}")

splits = {
    "train": TRAIN_DIR,
    "valid": VAL_DIR,
    "test": TEST_DIR,
}

# Setup device and model
device = get_device()

# Use eval transforms for all splits for consistency
_, eval_tf = make_transforms(IMG_SIZE)

# Load model
num_classes = len(datasets.ImageFolder(TRAIN_DIR).classes)
# Print detected classes (from train) for sanity
train_classes_preview = datasets.ImageFolder(TRAIN_DIR, transform=eval_tf).classes
print(f"Detected classes (train): {train_classes_preview} | num_classes: {num_classes}")

model = build_model(num_classes, feature_dim=512).to(device) 
#model = build_model(num_classes).to(device)
model.eval()
model.load_state_dict(torch.load(best_path, map_location=device))

# Get penultimate feature extractor (supports model.backbone or direct model)
backbone_holder = getattr(model, "backbone", model)
if not hasattr(backbone_holder, "features") or not hasattr(backbone_holder, "avgpool"):
    raise RuntimeError("Model does not expose 'features' and 'avgpool' modules required for feature extraction")

feature_extractor = torch.nn.Sequential(
    backbone_holder.features,
    backbone_holder.avgpool,
    torch.nn.Flatten(),
)

# Collect features/logits/labels/paths for all splits first
features_dict = {}
logits_dict = {}
labels_dict = {}
paths_dict = {}
classes_dict = {}  # track classes per split

for split, split_dir in splits.items():
    ds = datasets.ImageFolder(split_dir, transform=eval_tf)
    classes_dict[split] = ds.classes
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    features, logits, labels, paths = [], [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            feats = feature_extractor(imgs)
            logs = model(imgs)
            features.append(feats.cpu().numpy())
            logits.append(logs.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
            # append corresponding file paths
            paths.extend([ds.imgs[i][0] for i in range(len(labels)-len(lbls), len(labels))])
    features = np.concatenate(features, axis=0) if len(features) > 0 else np.zeros((0, 0))
    logits = np.concatenate(logits, axis=0) if len(logits) > 0 else np.zeros((0, 0))
    labels = np.array(labels, dtype=np.int64)

    #  original outputs (keep existing behavior)
    np.save(out_path("features", split, "npy"), features)
    np.save(out_path("logits", split, "npy"), logits)
    np.save(out_path("labels", split, "npy"), labels)
    with open(out_path("paths", split, "txt"), "w") as f:
        for p in paths:
            f.write(p + "\n")
    print(f"Saved {split} features/logits/labels/paths.")

    # Print label shape and class info for this split
    cls_counts = np.bincount(labels, minlength=num_classes)
    print(f"{split}: labels shape {labels.shape}; classes: {ds.classes}")
    print(f"{split}: label counts per class (by index): {cls_counts.tolist()}")

    features_dict[split] = features
    logits_dict[split] = logits
    labels_dict[split] = labels
    paths_dict[split] = paths

# --- PCA projection to 512 (fit on train, transform val/test) ---
# (removed padding and L2-normalization; PCA output is saved as-is)
# Use the features collected above
X_tr = features_dict.get("train", np.zeros((0, 0)))
X_val = features_dict.get("valid", np.zeros((0, 0)))
X_te  = features_dict.get("test",  np.zeros((0, 0)))

# --- Fit PCA on train features and project to 512 dims (or fewer if needed) ---
if X_tr.size > 0 and X_tr.ndim == 2:
    pca_dim = min(512, X_tr.shape[1])
    pca = PCA(n_components=pca_dim, svd_solver="auto", random_state=42)
    pca.fit(X_tr)
    joblib.dump(pca, os.path.join(NPY_DIR, f"pca_{pca_dim}.joblib"))

    Z_tr = pca.transform(X_tr)
    Z_val = pca.transform(X_val) if X_val.size else np.zeros((0, pca_dim))
    Z_te  = pca.transform(X_te)  if X_te.size  else np.zeros((0, pca_dim))

    np.save(out_path("features_pca", "train", "npy"), Z_tr)
    np.save(out_path("features_pca", "valid", "npy"), Z_val)
    np.save(out_path("features_pca", "test",  "npy"), Z_te)

    print(f"PCA saved: train {Z_tr.shape}, valid {Z_val.shape}, test {Z_te.shape}")
else:
    print("PCA skipped: empty features.")

# Summary
print("Summary:")
for split in ["train", "valid", "test"]:
    f = features_dict[split].shape
    l = logits_dict[split].shape
    n = labels_dict[split].shape
    p = len(paths_dict[split])
    print(f"  {split}: features {f}, logits {l}, labels {n}, paths {p}, classes {classes_dict.get(split)}")

# --- Convert labels to one-hot encoding ---
labels = np.load("npy_outputs/labels_train.npy")  # (N,)
num_classes = 4
labels_onehot = np.eye(num_classes, dtype=np.float32)[labels]  # (N, 4)

import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# =========================
# CONFIGURATION
# =========================

# Root of the dataset inside your project (relative to this file’s working dir)
ROOT = os.path.join(os.getcwd(), "data", "Data")

# Which split to visualize in the grid ("train", "test", or "valid")
SPLIT = "train"

# How many images to show per class in the grid
SAMPLES = 3

# Optional: make the random sampling reproducible
RANDOM_SEED = 42
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

#g =g.torchgenerator


# =========================
# HELPER FUNCTIONS
# =========================

def short_label(name: str) -> str:
    """
    Turn long folder names into shorter, readable labels for plotting.
    Adjust these rules to your taste; fall back to the raw name if no rule matches.
    """
    if name.startswith("adenocarcinoma"):
        return "adenocarcinoma"
    if name.startswith("squamous.cell.carcinoma"):
        return "squamous cell carcinoma"
    if name.startswith("large.cell.carcinoma"):
        return "large cell carcinoma"
    if name == "normal":
        return "normal"
    return name


def list_classes(split_path: str) -> list[str]:
    """
    Return a stable, sorted list of class directories for a given split.
    - Only directories
    - Ignore hidden entries (e.g., .DS_Store on macOS)
    """
    return sorted(
        d for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d)) and not d.startswith(".")
    )


def files_recursive(root: str) -> list[str]:
    """
    Collect all non-hidden files under `root` (at any depth).
    Using recursion avoids missing images stored in nested subfolders.
    """
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.startswith("."):
                paths.append(os.path.join(dirpath, f))
    return paths


def count_files_recursive(root: str) -> int:
    """
    Count image files under `root` (recursively), ignoring hidden files.
    Used for split/class counts.
    """
    total = 0
    for _, _, files in os.walk(root):
        total += sum(1 for f in files if not f.startswith("."))
    return total


# =========================
# QUICK SANITY CHECKS (optional)
# =========================

print("Top-level folders/files in ROOT:", os.listdir(ROOT))
split_path_for_grid = os.path.join(ROOT, SPLIT)
# Show the first directory we walk and how many files it has (just to confirm pathing)
for dirpath, _, filenames in os.walk(split_path_for_grid):
    print("First folder encountered in", SPLIT, "->", dirpath, ":", len(filenames), "files")
    break


# =========================
# COUNTS BY SPLIT AND CLASS
# =========================

for split in ["train", "test", "valid"]:
    split_path = os.path.join(ROOT, split)
    print(f"\n📂 {split.upper()} SET")
    if not os.path.isdir(split_path):
        print("  (missing)")
        continue

    classes = list_classes(split_path)
    total = 0
    for cls in classes:
        cls_path = os.path.join(split_path, cls)
        n = count_files_recursive(cls_path)
        total += n
        print(f"  {cls}: {n} images")
    print(f"  — Total: {total} images")


# =========================
# GRID PREVIEW FOR THE CHOSEN SPLIT
# =========================

# Rebuild classes specifically for the split we want to visualize
classes_for_grid = list_classes(split_path_for_grid)

rows, cols = len(classes_for_grid), SAMPLES
fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))

# Ensure `axes` is always 2D to make indexing simpler when rows == 1
if rows == 1:
    axes = [axes]

for r, cls in enumerate(classes_for_grid):
    cls_dir = os.path.join(split_path_for_grid, cls)
    files = files_recursive(cls_dir)

    if not files:
        # If a class has no images (or only hidden files), show a placeholder row
        for c in range(cols):
            ax = axes[r][c] if rows > 1 else axes[c]
            ax.text(0.5, 0.5, "No images found", ha="center", va="center")
            ax.set_title(short_label(cls) if c == 0 else "", fontsize=10)
            ax.axis("off")
        continue

    # Pick up to SAMPLES unique images; if fewer exist, pad with Nones
    picks = random.sample(files, min(SAMPLES, len(files)))
    picks += [None] * (SAMPLES - len(picks))

    for c, path in enumerate(picks):
        ax = axes[r][c] if rows > 1 else axes[c]
        if path is None:
            ax.axis("off")
            continue
        img = mpimg.imread(path)
        ax.imshow(img, cmap="gray")
        ax.set_title(short_label(cls) if c == 0 else "", fontsize=10)
        ax.axis("off")

plt.suptitle(f"{SAMPLES} random samples per class ({SPLIT.upper()} set)", fontsize=16)
plt.tight_layout()
plt.savefig("grid_preview.png", dpi=200, bbox_inches="tight")  # also save to disk
plt.show()


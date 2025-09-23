# Copilot Instructions for Cancer_Detections

## Project Overview
This repository is a CT-scan image classifier for lung cancer types, using PyTorch and torchvision. It supports advanced training features (ConvNeXt-Tiny backbone, label smoothing, balanced sampling, AdamW optimizer, cosine LR schedule, and test-time augmentation). Data is organized in a folder structure under `data/Data/{train,valid,test}/<class>/*.{png,jpg}`.

## Key Files & Structure
- `train_torch.py`: Main training script. Handles model setup, data loading, augmentation, training phases, and evaluation. Uses high-res images (IMG_SIZE=448) and supports MPS (Apple Silicon) and CUDA.
- `cancer_data_setup.py`: (Example) Downloads dataset from Kaggle using `kagglehub`. Not always required if data is present.
- `explore_dataset.py`: Visualizes dataset splits and samples per class. Defines conventions for class naming and plotting.
- `data/Data/`: Contains all image data, split into `train`, `valid`, and `test` folders, each with class subfolders.

## Developer Workflows
- **Training**: Run `train_torch.py` to train or fine-tune models. Adjust config variables at the top of the script for batch size, epochs, learning rates, etc.
- **Dataset Setup**: If data is missing, use `cancer_data_setup.py` to download from Kaggle. Otherwise, ensure `data/Data/` is populated.
- **Visualization**: Use `explore_dataset.py` to preview dataset splits and class distributions. Generates grid previews for sanity checks.
- **Model Checkpoints**: Trained models are saved as `.pt` files (e.g., `best_convnext_tiny.pt`, `best_resnet18.pt`).

## Patterns & Conventions
- **Class Names**: Class folders use descriptive names (e.g., `adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib`). Use `short_label()` in `explore_dataset.py` to map to readable labels.
- **Transforms**: All images are converted to 3-channel grayscale for compatibility with pretrained backbones.
- **Reproducibility**: Set `SEED` in `train_torch.py` for reproducible runs.
- **Device Selection**: `get_device()` auto-selects CUDA, MPS, or CPU.
- **Augmentation**: Custom CT-safe augmentations are used; see `make_transforms()` in `train_torch.py`.

## External Dependencies
- PyTorch, torchvision, scikit-learn, tqdm, numpy, matplotlib
- `kagglehub` (for dataset download)

## Example Commands
- Train model: `python train_torch.py`
- Download dataset: `python cancer_data_setup.py`
- Visualize dataset: `python explore_dataset.py`

## Integration Points
- No REST APIs or external services; all data is local or downloaded via Kaggle.
- Models and scripts are self-contained; outputs are saved locally.

## Tips for AI Agents
- Always check and update config variables at the top of scripts for custom runs.
- Use provided helper functions for class name mapping and data loading.
- Respect the folder structure for data and outputs.
- When adding new models or transforms, follow the patterns in `train_torch.py`.

---
*If any section is unclear or missing, please provide feedback to improve these instructions.*

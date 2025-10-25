import os, random, numpy as np, torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int | None, device: torch.device) -> None:
    if seed is None: return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)

def window_lung_tensor(t: torch.Tensor, level: float = -600., width: float = 1500.):
    hu = t * 4096.0 - 1024.0
    lo, hi = level - width / 2.0, level + width / 2.0
    hu = torch.clamp(hu, lo, hi)
    return (hu - lo) / max(1e-6, (hi - lo))

def make_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: window_lung_tensor(t)),
        transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
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

def get_loader_kwargs(device: torch.device):
    if device.type == "cuda":
        cpu_half = (os.cpu_count() or 2) // 2
        num_workers = min(8, max(2, cpu_half))
        return dict(num_workers=num_workers, pin_memory=True, persistent_workers=True)
    else:
        return dict(num_workers=0, pin_memory=False, persistent_workers=False)
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms.functional as TF
from typing import Tuple, List, Callable

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
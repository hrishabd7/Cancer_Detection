import os
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader

def get_datasets_and_loaders(train_dir, val_dir, test_dir, train_tf, eval_tf, batch, loader_kwargs, class_names):
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=eval_tf)
    test_ds  = datasets.ImageFolder(test_dir,  transform=eval_tf)

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
        train_ds, batch_size=batch, sampler=sampler, shuffle=False, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch, shuffle=False, **loader_kwargs
    )
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
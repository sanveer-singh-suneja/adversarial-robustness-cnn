"""
visualize.py — Standalone visualisation helpers

Run this after attack.py to generate extra plots from saved results,
or import individual functions into your own scripts.
"""

import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "./results"
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def denormalise(tensor):
    """Convert normalised tensor back to displayable [H,W,3] numpy array."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    img  = (tensor * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def plot_perturbation_magnified(original, perturbed, epsilon, save_path=None):
    """
    Show original | perturbation (magnified 10x) | perturbed side by side.
    Makes the imperceptible noise visually obvious.
    """
    perturbation = (perturbed - original) * 10   # magnify for visibility

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(denormalise(original));   axes[0].set_title("Original");              axes[0].axis("off")
    axes[1].imshow(denormalise(perturbation).clip(0,1)); axes[1].set_title(f"Noise × 10 (ε={epsilon})"); axes[1].axis("off")
    axes[2].imshow(denormalise(perturbed));  axes[2].set_title("Adversarial Example");   axes[2].axis("off")

    plt.suptitle("FGSM Perturbation Visualisation", fontsize=13)
    plt.tight_layout()

    path = save_path or os.path.join(RESULTS_DIR, f"perturbation_eps{epsilon}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_confidence_bars(logits_clean, logits_adv, true_label, save_path=None):
    """
    Bar chart comparing model confidence on clean vs adversarial input.
    Shows how the model's prediction distribution shifts after attack.
    """
    probs_clean = torch.softmax(torch.tensor(logits_clean), dim=0).numpy()
    probs_adv   = torch.softmax(torch.tensor(logits_adv),   dim=0).numpy()

    x   = np.arange(10)
    w   = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, probs_clean, w, label="Clean",       color="#2E4057", alpha=0.85)
    ax.bar(x + w/2, probs_adv,   w, label="Adversarial", color="#E63946", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=30, ha="right")
    ax.set_ylabel("Confidence")
    ax.set_title(f"Model Confidence Shift (True label: {CIFAR10_CLASSES[true_label]})")
    ax.legend()
    ax.axvline(true_label, color="green", linestyle="--", linewidth=1.5, label="True class")
    plt.tight_layout()

    path = save_path or os.path.join(RESULTS_DIR, "confidence_shift.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

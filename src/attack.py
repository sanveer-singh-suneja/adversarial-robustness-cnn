"""
attack.py — FGSM Adversarial Attack Implementation

Loads a pretrained ResNet18, applies FGSM perturbations at multiple
epsilon levels, and saves visualisations + accuracy results.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILONS    = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]   # perturbation strengths
BATCH_SIZE  = 64
NUM_WORKERS = 2
DATA_DIR    = "./data"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ImageNet normalisation constants (used by pretrained ResNet)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ── Data ─────────────────────────────────────────────────────────────────────
def get_loader():
    """Download CIFAR-10 test set and return a DataLoader."""
    transform = transforms.Compose([
        transforms.Resize(224),                        # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )


# ── Model ─────────────────────────────────────────────────────────────────────
def get_model():
    """Load or train ResNet18 on CIFAR-10, then return trained model."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(DEVICE)

    saved_path = "./results/models/cifar10_resnet18.pth"
    os.makedirs("./results/models", exist_ok=True)

    if os.path.exists(saved_path):
        print("Loading saved CIFAR-10 model ...")
        model.load_state_dict(torch.load(saved_path, map_location=DEVICE, weights_only=True))
        model.eval()
        return model

    print("Fine-tuning ResNet18 on CIFAR-10 (5 epochs) ...")
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        correct, total, total_loss = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            _, pred = out.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader):.3f} | Acc: {correct/total*100:.1f}%")

    torch.save(model.state_dict(), saved_path)
    print(f"  Model saved to {saved_path}")
    model.eval()
    return model


# ── FGSM ─────────────────────────────────────────────────────────────────────
def fgsm_attack(image: torch.Tensor, epsilon: float, gradient: torch.Tensor) -> torch.Tensor:
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014).

    Perturbed image = image + epsilon * sign(∇_x Loss)

    Args:
        image   : original input tensor  [B, C, H, W]
        epsilon : perturbation magnitude (0 = no attack)
        gradient: gradient of loss w.r.t. the input image

    Returns:
        Perturbed image clamped to [0, 1].
    """


    sign_grad = gradient.sign()
    perturbed = image + epsilon * sign_grad
    return perturbed  # no clamping — normalized tensors don't live in [0,1]


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, epsilon, criterion):
    """
    Run the model on all test batches after applying FGSM with given epsilon.

    Returns:
        accuracy (float) — fraction of correctly classified examples
        adversarial_examples (list) — a few (orig, perturbed, label, pred) for visualisation
    """
    correct           = 0
    total             = 0
    adversarial_store = []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if epsilon == 0:
            with torch.no_grad():
                outputs  = model(images)
                _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
            if len(adversarial_store) < 5:
                adversarial_store.append((
                    images[0].detach().cpu(),
                    images[0].detach().cpu(),
                    labels[0].item(),
                    predicted[0].item()
                ))
            continue

        images.requires_grad = True
        outputs = model(images)
        loss    = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = images.grad.data

        perturbed = fgsm_attack(images, epsilon, grad)

        with torch.no_grad():
            outputs_adv = model(perturbed)
        _, predicted = outputs_adv.max(1)

        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

        if len(adversarial_store) < 5:
            adversarial_store.append((
                images[0].detach().cpu(),
                perturbed[0].detach().cpu(),
                labels[0].item(),
                predicted[0].item()
            ))

    return correct / total, adversarial_store

# ── Visualisation ─────────────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def denormalise(tensor):
    """Reverse ImageNet normalisation for display."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    img  = tensor * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def plot_accuracy_curve(epsilons, accuracies):
    """Plot model accuracy vs epsilon — the key result of this study."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epsilons, accuracies, "o-", color="#2E4057", linewidth=2, markersize=8)
    ax.set_xlabel("Epsilon (perturbation strength)", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Adversarial Robustness: Accuracy vs Perturbation Strength (FGSM)", fontsize=13)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)
    for eps, acc in zip(epsilons, accuracies):
        ax.annotate(f"{acc:.2f}", (eps, acc), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "accuracy_vs_epsilon.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_examples(examples_per_eps, epsilons):
    """Show original vs perturbed images for the first few epsilon values."""
    n_eps  = min(4, len(epsilons))
    fig, axes = plt.subplots(n_eps, 2, figsize=(8, n_eps * 3))
    if n_eps == 1:
        axes = [axes]

    for row, eps_idx in enumerate(range(n_eps)):
        eps   = epsilons[eps_idx]
        orig, pert, label, pred = examples_per_eps[eps_idx][0]

        axes[row][0].imshow(denormalise(orig))
        axes[row][0].set_title(f"Original\nLabel: {CIFAR10_CLASSES[label]}", fontsize=10)
        axes[row][0].axis("off")

        axes[row][1].imshow(denormalise(pert))
        axes[row][1].set_title(
            f"Perturbed (ε={eps})\nPredicted: {CIFAR10_CLASSES[pred]}", fontsize=10
        )
        axes[row][1].axis("off")

    plt.suptitle("FGSM: Original vs Adversarial Examples", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "adversarial_examples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print("Loading data ...")
    loader    = get_loader()

    print("Loading pretrained ResNet18 ...")
    model     = get_model()
    criterion = nn.CrossEntropyLoss()

    accuracies       = []
    examples_per_eps = []

    for eps in EPSILONS:
        print(f"  Evaluating epsilon = {eps:.2f} ...")
        acc, examples = evaluate(model, loader, eps, criterion)
        accuracies.append(acc)
        examples_per_eps.append(examples)
        print(f"    Accuracy: {acc * 100:.2f}%")

    print("\nGenerating plots ...")
    plot_accuracy_curve(EPSILONS, accuracies)
    plot_examples(examples_per_eps, EPSILONS)

    # Print summary table
    print("\n── Summary ─────────────────────────────")
    print(f"{'Epsilon':>10}  {'Accuracy':>10}")
    print("─" * 25)
    for eps, acc in zip(EPSILONS, accuracies):
        print(f"{eps:>10.3f}  {acc * 100:>9.2f}%")
    print("─" * 25)
    print(f"Clean accuracy (ε=0):      {accuracies[0]*100:.2f}%")
    print(f"Accuracy at ε=0.3:         {accuracies[-1]*100:.2f}%")
    print(f"Accuracy drop:             {(accuracies[0]-accuracies[-1])*100:.2f}%")


if __name__ == "__main__":
    main()

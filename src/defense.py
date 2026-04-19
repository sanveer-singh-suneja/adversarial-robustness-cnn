"""
defense.py — Adversarial Training Defense

Trains a ResNet18 on a mix of clean + FGSM-perturbed images (adversarial training),
then evaluates it against the standard model to show the robustness-accuracy tradeoff.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS        = 5          # keep low for quick demonstration
BATCH_SIZE    = 64
LR            = 0.001
EPSILON_TRAIN = 0.1        # perturbation used during adversarial training
EVAL_EPSILONS = [0, 0.05, 0.1, 0.15, 0.2]
DATA_DIR      = "./data"
RESULTS_DIR   = "./results"
MODEL_DIR     = "./results/models"
os.makedirs(MODEL_DIR, exist_ok=True)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ── Data ─────────────────────────────────────────────────────────────────────
def get_loaders():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True,  download=True, transform=transform)
    test_set  = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = torch.utils.data.DataLoader(
        test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ── FGSM (inline for training) ────────────────────────────────────────────────
def fgsm(image, epsilon, gradient):
    return image + epsilon * gradient.sign()


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m.to(DEVICE)


# ── Training ──────────────────────────────────────────────────────────────────
def train_adversarial(model, loader, optimizer, criterion):
    """
    Adversarial training loop.
    Each batch: half clean examples + half FGSM-perturbed examples.
    This is the core defense strategy (Madry et al.).
    """
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # ── Step 1: generate adversarial examples ──
        images.requires_grad = True
        out  = model(images)
        loss = criterion(out, labels)
        model.zero_grad()
        loss.backward()
        perturbed = fgsm(images, EPSILON_TRAIN, images.grad.data).detach()
        images    = images.detach()

        # ── Step 2: train on 50% clean + 50% adversarial ──
        mixed  = torch.cat([images, perturbed], dim=0)
        mixed_labels = torch.cat([labels, labels], dim=0)

        optimizer.zero_grad()
        out_mixed = model(mixed)
        loss_mix  = criterion(out_mixed, mixed_labels)
        loss_mix.backward()
        optimizer.step()

        total_loss += loss_mix.item()

    return total_loss / len(loader)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, epsilon, criterion):
    model.eval()
    correct, total = 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if epsilon == 0:
            with torch.no_grad():
                pred = model(images).argmax(1)
            correct += pred.eq(labels).sum().item()
            total   += labels.size(0)
            continue

        images.requires_grad = True
        out  = model(images)
        loss = criterion(out, labels)
        model.zero_grad()
        loss.backward()
        perturbed = fgsm(images, epsilon, images.grad.data)

        with torch.no_grad():
            pred = model(perturbed).argmax(1)
        correct += pred.eq(labels).sum().item()
        total   += labels.size(0)

    return correct / total


# ── Comparison Plot ───────────────────────────────────────────────────────────
def plot_comparison(epsilons, std_accs, adv_accs):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epsilons, std_accs, "o--", color="#E63946", lw=2, label="Standard Model (no defense)")
    ax.plot(epsilons, adv_accs, "o-",  color="#2E4057", lw=2, label="Adversarially Trained Model")
    ax.set_xlabel("Epsilon (perturbation strength)", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Defense Comparison: Standard vs Adversarial Training", fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "defense_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    train_loader, test_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()

    # ── Standard model (no defense) ──
    print("\nEvaluating standard pretrained model ...")
    std_model = build_model()
    std_model.load_state_dict(
        torch.load("./results/models/cifar10_resnet18.pth", map_location=DEVICE, weights_only=True))
    std_accs  = []
    for eps in EVAL_EPSILONS:
        acc = evaluate(std_model, test_loader, eps, criterion)
        std_accs.append(acc)
        print(f"  ε={eps:.2f}  →  Accuracy: {acc*100:.2f}%")

    # ── Adversarially trained model ──
    print(f"\nAdversarial training for {EPOCHS} epochs (ε_train={EPSILON_TRAIN}) ...")
    adv_model   = build_model()
    optimizer   = optim.Adam(adv_model.parameters(), lr=LR)
    scheduler   = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    for epoch in range(1, EPOCHS + 1):
        loss = train_adversarial(adv_model, train_loader, optimizer, criterion)
        scheduler.step()
        print(f"  Epoch {epoch}/{EPOCHS}  |  Loss: {loss:.4f}")

    torch.save(adv_model.state_dict(), os.path.join(MODEL_DIR, "adv_trained_resnet18.pth"))
    print("  Model saved.")

    print("\nEvaluating adversarially trained model ...")
    adv_accs = []
    for eps in EVAL_EPSILONS:
        acc = evaluate(adv_model, test_loader, eps, criterion)
        adv_accs.append(acc)
        print(f"  ε={eps:.2f}  →  Accuracy: {acc*100:.2f}%")

    # ── Summary ──
    print("\n── Comparison Summary ───────────────────────────────")
    print(f"{'Epsilon':>8}  {'Standard':>10}  {'Adversarial':>12}")
    print("─" * 36)
    for eps, s, a in zip(EVAL_EPSILONS, std_accs, adv_accs):
        print(f"{eps:>8.2f}  {s*100:>9.2f}%  {a*100:>11.2f}%")

    plot_comparison(EVAL_EPSILONS, std_accs, adv_accs)
    print("\nDone. Check the results/ folder for plots.")


if __name__ == "__main__":
    main()

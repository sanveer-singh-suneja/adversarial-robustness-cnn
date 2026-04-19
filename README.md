# Adversarial Robustness Study — CNN Image Classifier

A study of adversarial vulnerability and defense in deep neural networks, implemented in PyTorch on CIFAR-10.

Implements the **Fast Gradient Sign Method (FGSM)** attack (Goodfellow et al., 2014) and **Adversarial Training** as a defense strategy, analysing the accuracy-robustness tradeoff across perturbation strengths.

---

## Motivation

Modern computer vision models are surprisingly fragile — a barely perceptible pixel-level perturbation can flip a model's prediction with high confidence. This project studies:

1. **How vulnerable** a standard ResNet18 is to FGSM attacks across varying epsilon (ε) values
2. **How adversarial training** recovers robustness, and at what cost to clean accuracy

This is directly motivated by **Trustworthy AI** research — building vision systems that are reliable and safe in real-world, adversarial conditions.

---

## Results

| Epsilon (ε) | Standard Model | Adversarially Trained |
|:-----------:|:--------------:|:---------------------:|
| 0.00        | 89.22%         | 92.97%                |
| 0.05        | 6.62%          | 87.01%                |
| 0.10        | 4.94%          | 78.39%                |
| 0.15        | 6.64%          | 78.75%                |
| 0.20        | 9.01%          | 74.44%                |

**Key finding:** The standard model collapses from 89.22% to under 7% accuracy under an FGSM attack at ε=0.05 — a perturbation completely invisible to the human eye. Adversarial training not only recovers this robustness (87.01% at ε=0.05) but also improves clean accuracy to 92.97%, demonstrating that training on adversarial examples forces the model to learn more generalised, robust features rather than brittle texture shortcuts.

---

## Project Structure

```
adversarial-robustness-cnn/
├── src/
│   ├── attack.py       # FGSM attack — evaluate accuracy across epsilon values
│   ├── defense.py      # Adversarial training defense + comparison plots
│   └── visualize.py    # Visualisation utilities (perturbation, confidence bars)
├── results/            # Output plots saved here (gitignored)
│   └── models/         # Saved model checkpoints (gitignored)
├── data/               # CIFAR-10 auto-downloaded here (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/sanveer-singh-suneja/adversarial-robustness-cnn
cd adversarial-robustness-cnn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run FGSM attack (downloads CIFAR-10 automatically, ~170MB)
python src/attack.py

# 4. Run adversarial training defense
python src/defense.py
```

Results and plots will be saved to `results/`.

---

## Key Concepts

**FGSM (Fast Gradient Sign Method)**
Computes the gradient of the loss with respect to the *input image* (not model weights), then perturbs the image in the direction that maximises the loss:

```
x_adv = x + ε · sign(∇_x L(θ, x, y))
```

Where ε controls the perturbation magnitude. Small ε → imperceptible to humans, still fools the model.

**Adversarial Training**
Trains the model on a mix of clean and adversarial examples each batch. Forces the model to learn robust features rather than brittle texture shortcuts.

---

## References

- Goodfellow et al. (2014) — *Explaining and Harnessing Adversarial Examples* — [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)
- Madry et al. (2017) — *Towards Deep Learning Models Resistant to Adversarial Attacks* — [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)
- PyTorch Official FGSM Tutorial — [pytorch.org/tutorials](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)

---

## Tech Stack

`Python` · `PyTorch` · `Torchvision` · `CIFAR-10` · `ResNet18` · `Matplotlib`

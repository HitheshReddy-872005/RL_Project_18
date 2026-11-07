# Project 18: Data-Augmentation Policy Search with REINFORCE

---

## Slide 1: Motivation & Problem Statement

- Data augmentation is one of the cheapest ways to improve generalisation, but good policies are usually **hand-designed** and depend on the dataset.
- AutoAugment (Cubuk et al., 2019) showed augmentation can be **learned with RL**, but it used thousands of GPU-hours.
- **Question:** can a small REINFORCE controller find useful CIFAR-10 sub-policies on a **single CPU in under 30 minutes**? How does it compare with random search and the standard flip+crop heuristic?
- **Setting:** CIFAR-10, a 2,000-image proxy training set, a 1,000-image validation set, a 3-epoch proxy CNN, and 60 search episodes.

---

## Slide 2: RL Formulation & Search Space Design

- **Action:** a sub-policy $a = (o_1, p_1, m_1, o_2, p_2, m_2)$.
  - 8 ops: ShearX, ShearY, Rotate, Solarize, Posterize, Contrast, Brightness, Sharpness.
  - $p \in \{0.1,\dots,1.0\}$, $m \in \{0,\dots,9\}$, giving $|\mathcal{A}| = 640{,}000$.
- **State:** one-hot encoding of the previous sub-policy, plus the previous reward and the EMA baseline (58 dims).
- **Transition:** re-initialise the proxy CNN with a fixed seed, train 3 epochs with augmentation $a_t$, evaluate. Fixed seeds (common random numbers) mean only the augmentation affects the reward.
- **Reward:** top-1 accuracy of the proxy on the held-out validation subset, $R_t \in [0,1]$.
- **Structure:** effectively a contextual bandit, because the proxy is reset every episode.
- Full write-up: `docs/problem_formulation.md`.

---

## Slide 3: Controller Architecture & REINFORCE Algorithm

- **PolicyController:** MLP (58 → 64 → 64, tanh) with six categorical heads (op1, prob1, mag1, op2, prob2, mag2).
  - Heads are zero-initialised, so the controller starts as an exactly uniform policy (identical to random search).
- **Update:** $\mathcal{L} = -\log\pi_\theta(a\mid s)\,\hat{A} - \lambda_H \mathcal{H}[\pi_\theta]$.
  - Advantage $\hat{A} = (R - b)/\sigma$, clipped to ±3, where $b$ is an EMA baseline ($\beta=0.9$) and $\sigma$ an EMA standard deviation.
  - Optimiser: Adam, lr 0.02, $\lambda_H = 0.01$.
- **Loop per episode:** sample → train proxy (~4–6 s on CPU) → reward → policy-gradient step → log to CSV.
- **Proxy:** SmallProxyNet with 3× [Conv → BN → ReLU → MaxPool] and a linear head, trained with AdamW + OneCycle.

---

## Slide 4: Empirical Results & Baseline Comparison

| Method (proxy, 60 episodes) | Initial Acc (first 5) | Final Acc (last 5) | Peak Acc |
|---|---|---|---|
| REINFORCE controller | 0.437 | **0.479** | 0.487 |
| Uniform random search | 0.448 | 0.452 | **0.489** |
| Heuristic flip+crop (5 seeds) | — | 0.419 ± 0.006 | 0.428 |
| No augmentation (5 seeds) | — | 0.470 ± 0.012 | 0.489 |

- **REINFORCE vs. random:** REINFORCE moves the *average* sampled policy from harmful to helpful (final 0.479 vs. 0.452 for random). Its *peak* is no better than random search.
- **Best RL sub-policy:** (ShearY, p=0.9, m=2) → (Contrast, p=0.9, m=1). The controller learned to prefer **mild, low-magnitude** geometric and photometric ops.
- **Target validation** (ConvNet, 10k images, 20 epochs, full 10k CIFAR-10 test set):
  - flip+crop: **79.93%**
  - flip+crop + RL policy: 79.37% (−0.56 pp)
  - flip+crop + random-search policy: 79.33% (−0.60 pp)
- **Takeaway:** in a 3-epoch proxy, strong augmentation is penalised (even flip+crop hurts, because it underfits). The proxy reward favours "almost no augmentation", and that does not transfer to the longer target training.
- Figures: `results/learning_curve.png`, `results/policy_distribution.png`.

---

## Slide 5: Limitations, Compute Footprint, and Safety/Ethics

**Limitations**
- **Proxy–target mismatch:** a 3-epoch / 2k-image reward measures how fast the model learns early, not how well it regularises. The best proxy policies did not beat flip+crop on the target.
- **Premature convergence:** with lr 0.02 the controller's entropy collapsed from 13.4 to below 0.4 by episode 25 (about 0.02 by episode 60), and it ended up cycling between two policies.
  - Ablation (lr 0.005, $\lambda_H$ = 0.05) kept entropy at ~12.7 and found a higher peak (0.503), but its final mean was lower (0.459). This is a clear exploration–exploitation trade-off.
- **Statistical weakness:** single search seed, 60 noisy samples, and differences of 1–3 pp, which is close to the seed std of the no-augmentation baseline (1.2 pp).
- **Small space:** a single sub-policy instead of AutoAugment's 5×2 policy, and no Cutout, flips, or colour ops.

**Compute footprint**
- CPU only (8 threads): about 27 min total. That covers 130 proxy trainings at ~5 s each and 3 target trainings at ~5–6 min each.
- AutoAugment used about 5,000 GPU-hours. Our run is roughly 10⁴× cheaper, at the cost of reward fidelity.

**Safety / ethics**
- A learned augmentation can encode dataset biases (for example, photometric ops that hurt darker or low-contrast images). Policies should be audited per class and per subgroup.
- An over-fitted search can overfit the validation subset and give a false sense of robustness. Always report on a held-out test set, as done here.
- Energy use grows quickly as search budgets increase. Prefer cheap proxies, shared search results, or search-free methods (RandAugment / TrivialAugment).

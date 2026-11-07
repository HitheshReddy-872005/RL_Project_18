# Problem Formulation: Data-Augmentation Policy Search as Reinforcement Learning

## 1. Overview

We want an augmentation sub-policy that gives the best generalisation on CIFAR-10. Following AutoAugment, we treat this as a sequential decision problem. A stochastic **controller** $\pi_\theta$ proposes a sub-policy. The **environment** trains a proxy CNN from scratch using that sub-policy and returns the proxy's validation accuracy as the reward. The controller is updated with REINFORCE using an exponential-moving-average (EMA) baseline.

Each episode is one proposal → train → evaluate cycle. The search runs for $T = 60$ episodes.

## 2. Search Space (Action Space)

**Operations** $\mathcal{O}$ (8 operations, PIL-based). The magnitude level $M \in \{0,\dots,9\}$ is mapped linearly to the intensity ranges below:

| Operation | Intensity at level $M$ | Random sign |
|---|---|---|
| ShearX | shear $= 0.3 \cdot M/9$ | yes |
| ShearY | shear $= 0.3 \cdot M/9$ | yes |
| Rotate | angle $= 30^\circ \cdot M/9$ | yes |
| Solarize | threshold $= 256 - 256 \cdot M/9$ | no |
| Posterize | bits $= 8 - \mathrm{round}(4 \cdot M/9)$ | no |
| Contrast | factor $= 1 \pm 0.9 \cdot M/9$ | yes |
| Brightness | factor $= 1 \pm 0.9 \cdot M/9$ | yes |
| Sharpness | factor $= 1 \pm 0.9 \cdot M/9$ | yes |

**Probabilities** $\mathcal{P} = \{0.1, 0.2, \dots, 1.0\}$, **magnitudes** $\mathcal{M} = \{0, 1, \dots, 9\}$.

An action is one **sub-policy**: a 6-tuple of discrete choices, applied in order to each training image:

$$
a = (o_1, p_1, m_1, o_2, p_2, m_2) \in \mathcal{A} = (\mathcal{O}\times\mathcal{P}\times\mathcal{M})^2,
\qquad |\mathcal{A}| = (8 \cdot 10 \cdot 10)^2 = 640{,}000.
$$

For each image, operation $o_i$ is applied with probability $p_i$ at magnitude $m_i$, and $o_1$ always goes before $o_2$. Augmentation is redrawn every epoch, so the training distribution is stochastic.

## 3. State Space

The controller is conditioned on a compact summary of the previous step:

$$
s_t = \big[\,\mathrm{onehot}(a_{t-1}),\; R_{t-1},\; b_{t-1}\,\big] \in \mathbb{R}^{56+2}
$$

- $\mathrm{onehot}(a_{t-1})$: the six one-hot encodings of the previous sub-policy ($8+10+10+8+10+10 = 56$ dims). All zeros at $t=1$.
- $R_{t-1}$: the previous reward. At $t=1$ this is the no-augmentation reference accuracy.
- $b_{t-1}$: the current EMA baseline.

The proxy network's weights are re-initialised every episode, so nothing about the proxy is carried between steps. That makes the problem close to a **contextual bandit**: the state gives the controller context, but the action does not change the proxy's future dynamics.

## 4. Policy (Controller)

`PolicyController`: an MLP with layers 58 → 64 → 64 and tanh activations, plus six independent categorical heads. The head weights are zero-initialised, so the starting policy is exactly uniform, the same as the random-search baseline.

$$
\pi_\theta(a \mid s) = \prod_{k \in \{o_1,p_1,m_1,o_2,p_2,m_2\}} \mathrm{Cat}\big(a_k \mid \mathrm{softmax}(W_k h_\theta(s) + c_k)\big),
\qquad
\log \pi_\theta(a\mid s) = \sum_k \log \pi_\theta^{(k)}(a_k \mid s).
$$

## 5. Environment Transition Dynamics

Given $a_t$, the environment does the following:

1. Re-initialise `SmallProxyNet` (3× [Conv3×3 → BN → ReLU → MaxPool] → Linear) with a fixed seed. Using the same seed every episode (common random numbers) removes initialisation noise from reward comparisons.
2. Train for 3 epochs on a fixed 2,000-image CIFAR-10 training subset, augmented with sub-policy $a_t$ (AdamW, OneCycle LR, batch 128).
3. Evaluate top-1 accuracy on a disjoint 1,000-image validation subset (no augmentation).
4. Emit reward $R_t$, update the baseline, and build $s_{t+1}$ from $(a_t, R_t, b_t)$.

Formally, $s_{t+1} = f(s_t, a_t, R_t)$ is deterministic given the reward. The reward itself is stochastic because augmentation sampling is random, so $P(s_{t+1}\mid s_t,a_t)$ is induced by $R_t \sim p(R \mid a_t)$. The episode horizon is a single decision; the search is $T$ such decisions.

## 6. Reward Function

$$
R_t = \mathrm{Acc}_{\text{val}}\big(\,\mathrm{Train}(f_{\phi_0};\ \mathcal{D}_{\text{train}},\ a_t,\ 3\ \text{epochs})\,\big) \in [0, 1]
$$

The reward is the top-1 accuracy of the freshly trained proxy on the held-out validation subset. The objective is $J(\theta) = \mathbb{E}_{a \sim \pi_\theta}[R(a)]$.

## 7. Learning Rule (REINFORCE with EMA Baseline)

EMA baseline (decay $\beta = 0.9$, initialised to the no-augmentation reference reward):

$$
b_{t} = \beta\, b_{t-1} + (1-\beta) R_t
$$

Policy-gradient loss, with a small entropy bonus to keep exploring:

$$
\mathcal{L}(\theta) = -\log \pi_\theta(a_t \mid s_t)\cdot \hat{A}_t \;-\; \lambda_H \,\mathcal{H}\big[\pi_\theta(\cdot\mid s_t)\big],
\qquad \hat{A}_t = \mathrm{clip}\!\left(\frac{R_t - b_{t-1}}{\sigma_{t-1}}, -3, 3\right)
$$

$\sigma_t$ is an EMA of the advantage's standard deviation. Rewards only differ by about 1–3 percentage points, so this rescales the core term $-\log\pi(a|s)\,(R-b)$ to unit magnitude. It does not change the direction of the gradient. Parameters are updated with Adam (lr $= 0.02$), with $\lambda_H = 0.01$.

## 8. Baselines

| Baseline | Description |
|---|---|
| Heuristic | Random horizontal flip + random crop (4 px padding). Proxy trained over 5 seeds; report mean ± std. |
| No augmentation | Same proxy and seeds, no augmentation. Used as the reference reward. |
| Uniform random search | Same 60-episode loop, with sub-policies sampled uniformly and no controller updates. |
| Target validation | Best RL sub-policy (stacked on flip+crop) vs. flip+crop alone. Uses a larger ConvNet (ResNet-18 when a GPU is present), 10,000 training images, 20 epochs, evaluated on the full 10,000-image CIFAR-10 test set. |


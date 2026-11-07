# Results Summary

## Search performance (initial = mean of first 5 episodes, final = mean of last 5 episodes)

| Method | Episodes | Initial Acc | Final Acc | Mean Acc | Peak Acc |
|---|---|---|---|---|---|
| REINFORCE controller | 60 | 0.4368 | 0.4788 | 0.4722 | 0.4870 |
| Uniform random search | 60 | 0.4476 | 0.4516 | 0.4451 | 0.4890 |
| Heuristic flip+crop (5 seeds) | - | - | - | 0.4194 ± 0.0061 | 0.4280 |
| No augmentation (5 seeds) | - | - | - | 0.4698 ± 0.0124 | 0.4890 |

## Peak proxy accuracy

| Heuristic | Random Search | RL Controller |
|---|---|---|
| 0.4280 | 0.4890 | 0.4870 |

## Best sub-policy found

| Source | Operation 1 | Prob 1 | Mag 1 | Operation 2 | Prob 2 | Mag 2 | Proxy Acc |
|---|---|---|---|---|---|---|---|
| REINFORCE (best episode) | ShearY | 0.9 | 2 | Contrast | 0.9 | 1 | 0.4870 |
| REINFORCE (final greedy) | ShearY | 0.3 | 0 | Contrast | 0.3 | 1 | n/a |
| Random search (best episode) | Contrast | 0.7 | 0 | Posterize | 0.9 | 4 | 0.4890 |

## Target model validation (convnet, 10000 train images, 20 epochs, 10000 CIFAR-10 test images)

| Configuration | Test Accuracy | Δ vs Heuristic | Train Time (s) |
|---|---|---|---|
| heuristic (flip+crop) | 0.7993 | +0.00 pp | 306.7 |
| heuristic + rl best policy | 0.7937 | -0.56 pp | 346.5 |
| heuristic + random best policy | 0.7933 | -0.60 pp | 365.1 |

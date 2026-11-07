# Project 18: Data-Augmentation Policy Search (REINFORCE on CIFAR-10)

## Setup
```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Run
```powershell
.venv/Scripts/python.exe src/main.py
.venv/Scripts/python.exe src/plot_results.py
```
`main.py` runs four stages: the heuristic baseline (5 seeds), the REINFORCE search (60 episodes), the uniform random search (60 episodes), and target validation. Use `--modes` to run individual stages and `--help` to see all hyperparameters.

## Layout
- `src/main.py`: search space, `SmallProxyNet`, `PolicyController`, REINFORCE loop, baselines, target validation
- `src/plot_results.py`: `learning_curve.png`, `policy_distribution.png`, `summary.md`
- `results/`: CSV logs, JSON results, `controller.pt`, plots, `run_log.txt`, and `ablation_rl_stabilized/`
- `docs/`: `problem_formulation.md`, `presentation_slides.md`, `results_summary.md`


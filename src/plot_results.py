import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POLICY_KEYS = ("op1", "prob1", "mag1", "op2", "prob2", "mag2")


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def rolling_mean(values, window):
    return np.array([np.mean(values[max(0, i - window + 1):i + 1]) for i in range(len(values))])


def rewards_of(rows):
    return np.array([float(row["reward"]) for row in rows])


def plot_learning_curve(rl_rows, random_rows, heuristic, output, window):
    fig, ax = plt.subplots(figsize=(10, 6))
    for rows, label, color in ((rl_rows, "REINFORCE controller", "tab:blue"), (random_rows, "Uniform random search", "tab:orange")):
        if not rows:
            continue
        episodes = np.array([int(row["episode"]) for row in rows])
        rewards = rewards_of(rows)
        ax.plot(episodes, rewards, "o", color=color, alpha=0.3, markersize=4)
        ax.plot(episodes, rolling_mean(rewards, window), "-", color=color, linewidth=2.2, label=f"{label} ({window}-episode rolling mean)")
        ax.plot(episodes, np.maximum.accumulate(rewards), ":", color=color, linewidth=1.4, label=f"{label} (best so far)")
    if heuristic:
        mean, std = heuristic["heuristic_mean"], heuristic["heuristic_std"]
        ax.axhline(mean, color="tab:green", linestyle="--", linewidth=2, label=f"Heuristic flip+crop ({mean:.3f} ± {std:.3f}, 5 seeds)")
        ax.axhspan(mean - std, mean + std, color="tab:green", alpha=0.12)
        ax.axhline(heuristic["no_aug_mean"], color="gray", linestyle="-.", linewidth=1.5, label=f"No augmentation ({heuristic['no_aug_mean']:.3f})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Proxy validation top-1 accuracy (reward)")
    ax.set_title("Augmentation policy search on CIFAR-10: REINFORCE vs. random search")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def operation_counts(rows, last_n):
    counter = Counter()
    for row in rows[-last_n:]:
        counter[row["op1"]] += 1
        counter[row["op2"]] += 1
    return counter


def plot_policy_distribution(rl_rows, random_rows, rl_best, output, last_n):
    rl_counts = operation_counts(rl_rows, last_n)
    random_counts = operation_counts(random_rows, last_n)
    names = (rl_best or {}).get("operation_names") or sorted(set(rl_counts) | set(random_counts))
    names = sorted(names, key=lambda name: (-rl_counts.get(name, 0), name))
    x = np.arange(len(names))
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, [rl_counts.get(name, 0) for name in names], width, label="REINFORCE controller", color="tab:blue")
    ax.bar(x + width / 2, [random_counts.get(name, 0) for name in names], width, label="Uniform random search", color="tab:orange")
    expected = 2 * min(last_n, max(len(rl_rows), 1)) / max(len(names), 1)
    ax.axhline(expected, color="gray", linestyle="--", linewidth=1, label=f"Uniform expectation ({expected:.1f})")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(f"Selections in final {last_n} episodes (op1 + op2)")
    ax.set_title(f"Most frequently selected augmentation operations (final {last_n} episodes)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def fmt(value):
    return "n/a" if value is None else f"{value:.4f}"


def search_stats(rows, k):
    if not rows:
        return None
    rewards = rewards_of(rows)
    return {"initial": float(rewards[:k].mean()), "final": float(rewards[-k:].mean()), "peak": float(rewards.max()), "mean": float(rewards.mean()), "episodes": len(rows)}


def policy_row(label, policy, reward):
    if not policy:
        return f"| {label} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    return f"| {label} | {policy['op1']} | {policy['prob1']} | {policy['mag1']} | {policy['op2']} | {policy['prob2']} | {policy['mag2']} | {fmt(reward)} |"


def build_summary(rl_rows, random_rows, heuristic, rl_best, random_best, target, k):
    rl_stats, random_stats = search_stats(rl_rows, k), search_stats(random_rows, k)
    lines = ["# Results Summary", "", f"## Search performance (initial = mean of first {k} episodes, final = mean of last {k} episodes)", ""]
    lines += ["| Method | Episodes | Initial Acc | Final Acc | Mean Acc | Peak Acc |", "|---|---|---|---|---|---|"]
    for label, stats in (("REINFORCE controller", rl_stats), ("Uniform random search", random_stats)):
        if stats:
            lines.append(f"| {label} | {stats['episodes']} | {fmt(stats['initial'])} | {fmt(stats['final'])} | {fmt(stats['mean'])} | {fmt(stats['peak'])} |")
    if heuristic:
        accs = heuristic["heuristic_accuracies"]
        lines.append(f"| Heuristic flip+crop ({len(accs)} seeds) | - | - | - | {fmt(heuristic['heuristic_mean'])} ± {heuristic['heuristic_std']:.4f} | {fmt(max(accs))} |")
        lines.append(f"| No augmentation ({len(heuristic['no_aug_accuracies'])} seeds) | - | - | - | {fmt(heuristic['no_aug_mean'])} ± {heuristic['no_aug_std']:.4f} | {fmt(max(heuristic['no_aug_accuracies']))} |")
    lines += ["", "## Peak proxy accuracy", "", "| Heuristic | Random Search | RL Controller |", "|---|---|---|"]
    lines.append(f"| {fmt(max(heuristic['heuristic_accuracies']) if heuristic else None)} | {fmt(random_stats['peak'] if random_stats else None)} | {fmt(rl_stats['peak'] if rl_stats else None)} |")
    lines += ["", "## Best sub-policy found", "", "| Source | Operation 1 | Prob 1 | Mag 1 | Operation 2 | Prob 2 | Mag 2 | Proxy Acc |", "|---|---|---|---|---|---|---|---|"]
    lines.append(policy_row("REINFORCE (best episode)", rl_best and rl_best["best_policy"], rl_best and rl_best["best_reward"]))
    if rl_best and rl_best.get("greedy_policy"):
        lines.append(policy_row("REINFORCE (final greedy)", rl_best["greedy_policy"], None))
    lines.append(policy_row("Random search (best episode)", random_best and random_best["best_policy"], random_best and random_best["best_reward"]))
    if target:
        lines += ["", f"## Target model validation ({target['arch']}, {target['train_size']} train images, {target['epochs']} epochs, {target['test_size']} CIFAR-10 test images)", ""]
        lines += ["| Configuration | Test Accuracy | Δ vs Heuristic | Train Time (s) |", "|---|---|---|---|"]
        reference = target["results"][0]["test_accuracy"]
        for result in target["results"]:
            delta = result["test_accuracy"] - reference
            lines.append(f"| {result['config']} | {fmt(result['test_accuracy'])} | {delta * 100:+.2f} pp | {result['seconds']} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Plot augmentation search results")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--docs-dir", default=str(ROOT / "docs"))
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--last-n", type=int, default=20)
    args = parser.parse_args()
    results_dir, docs_dir = Path(args.results_dir), Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    rl_rows = read_csv(results_dir / "rl_search_log.csv")
    random_rows = read_csv(results_dir / "random_search_log.csv")
    heuristic = read_json(results_dir / "heuristic_baseline.json")
    rl_best = read_json(results_dir / "rl_best_policy.json")
    random_best = read_json(results_dir / "random_best_policy.json")
    target = read_json(results_dir / "target_validation.json")
    plot_learning_curve(rl_rows, random_rows, heuristic, results_dir / "learning_curve.png", args.window)
    plot_policy_distribution(rl_rows, random_rows, rl_best, results_dir / "policy_distribution.png", args.last_n)
    summary = build_summary(rl_rows, random_rows, heuristic, rl_best, random_best, target, args.window)
    (results_dir / "summary.md").write_text(summary, encoding="utf-8")
    (docs_dir / "results_summary.md").write_text(summary, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()


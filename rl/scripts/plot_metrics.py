"""Plot GRPO/SFT JSONL metrics written by training scripts.

Example:
    python scripts/plot_metrics.py --in outputs/qwen3_4b_grpo/metrics.jsonl \
        --out outputs/qwen3_4b_grpo/metrics.png
"""
import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no metrics found in {args.inp}")

    def series(key):
        xs, ys = [], []
        for row in rows:
            if key not in row:
                continue
            try:
                xs.append(int(row.get("step", len(xs))))
                ys.append(float(row[key]))
            except (TypeError, ValueError):
                continue
        return xs, ys

    plots = [
        ("reward", "reward"),
        ("reward_std", "reward std"),
        ("frac_reward_zero_std", "zero-std frac"),
        ("completions/mean_length", "mean completion len"),
        ("completions/clipped_ratio", "clipped ratio"),
        ("loss", "loss"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for ax, (key, title) in zip(axes.flatten(), plots):
        xs, ys = series(key)
        ax.set_title(title)
        if xs:
            ax.plot(xs, ys)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

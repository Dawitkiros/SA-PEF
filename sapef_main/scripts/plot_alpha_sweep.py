"""Plot α-sensitivity ablation.

For each dataset block in the manifest, draws one figure with all α values
(α=0.0 labelled as EF, α=1.0 as SAEF, in-between as SA-PEF). Mean accuracy
across seed CSVs, Gaussian-smoothed, figsize 3.35×2.3 in.
"""
from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from plot_utils import (
    ALPHA_STYLE,
    accuracy_column,
    apply_plot_style,
    expand_configs,
    gaussian_smoothing,
    load_result_csv,
    new_axes,
    read_config,
    resolve_paths,
    save_and_close,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/figures.yaml")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--smooth", type=float, default=20.0,
                        help="Gaussian-smoothing sigma")
    parser.add_argument("--markevery", type=int, default=60)
    return parser.parse_args()


def _alpha_from_config(config_path: Path) -> float:
    cfg = read_config(config_path)
    return float(cfg.get("alpha-r", cfg.get("alpha_r", 0.0)))


def _style_for_alpha(alpha: float) -> dict:
    """Pick the closest 0.1-step bucket; falls back gracefully."""
    rounded = round(alpha, 1)
    if rounded in ALPHA_STYLE:
        return ALPHA_STYLE[rounded]
    return {"color": None, "marker": "x", "label_prefix": "SA-PEF"}


def _label(alpha: float, style: dict) -> str:
    return f"{style['label_prefix']} " + r"($\alpha_r=" + f"{alpha:.1f}" + r"$)"


def plot_dataset(name: str, patterns: list[str], out_dir: Path,
                 smooth: float, markevery: int) -> None:
    apply_plot_style()
    fig, ax = new_axes(figsize=(3.35, 2.3))
    plotted = 0

    for config_path in sorted(expand_configs(patterns), key=_alpha_from_config):
        alpha = _alpha_from_config(config_path)
        try:
            df = load_result_csv(config_path)
        except FileNotFoundError as exc:
            print(f"[warn] {config_path}: {exc}")
            continue

        df = df[df["round"] > 0].copy().sort_values("round").reset_index(drop=True)
        if df.empty:
            continue

        acc = df[accuracy_column(df)].astype(float)
        if acc.max() <= 1.0:
            acc = acc * 100.0

        style = _style_for_alpha(alpha)
        ax.plot(
            df["round"].to_numpy(),
            gaussian_smoothing(acc.to_numpy(), sigma=smooth),
            color=style["color"],
            marker=style["marker"],
            lw=1.0 if alpha != 0.8 else 1.5,  # accent default α=0.85 bucket
            label=_label(alpha, style),
            markevery=markevery,
        )
        plotted += 1

    if plotted == 0:
        print(f"[skip] {name}: no readable logs")
        plt.close(fig)
        return

    ax.set_xlabel("Number of rounds")
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(
        frameon=False,
        ncol=2,
        fontsize=5.5,
        handlelength=1.0,
        handletextpad=0.2,
        labelspacing=0.2,
        columnspacing=0.7,
        markerscale=0.7,
        loc="lower right",
        borderaxespad=0.2,
    )
    save_and_close(fig, out_dir, f"alpha_sweep_{name}.pdf")


def main() -> None:
    args = parse_args()
    _, manifest, out_dir = resolve_paths(args)
    entry = manifest.get("figures", {}).get("alpha_ablation", {})
    datasets = entry.get("datasets", {
        "cifar10": ["configs/alpha_sweep/cifar10/*.toml"],
        "cifar100": ["configs/alpha_sweep/cifar100/*.toml"],
    })
    for dataset_name, patterns in datasets.items():
        plot_dataset(dataset_name, patterns, out_dir, args.smooth, args.markevery)


if __name__ == "__main__":
    main()

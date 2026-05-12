"""Plot gradient-mismatch diagnostics for EF, SAEF, SA-PEF, CSER."""
from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import pandas as pd

from plot_utils import (
    apply_plot_style,
    expand_configs,
    gaussian_smoothing,
    load_result_csv,
    method_color,
    method_label,
    method_sort_key,
    new_axes,
    read_config,
    resolve_paths,
    save_and_close,
)


PLOT_METHODS = {"ef", "saef", "sapef", "cser"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/figures.yaml")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--markevery", type=int, default=25)
    parser.add_argument("--smooth", type=float, default=5.0,
                        help="Gaussian smoothing sigma applied in log-space")
    return parser.parse_args()


def _method_from_config(config_path: Path) -> str:
    cfg = read_config(config_path)
    return str(cfg.get("algorithm", {}).get("name", config_path.stem)).lower()


def _load_mismatch(config_path: Path) -> pd.DataFrame | None:
    try:
        df = load_result_csv(config_path)
    except FileNotFoundError:
        return None
    for col in ("fit_grad_mismatch_sq", "grad_mismatch_sq"):
        if col in df.columns:
            return df[["round", col]].rename(columns={col: "grad_mismatch_sq"})
    return None


def plot_dataset(name: str, patterns: list[str], out_dir: Path, markevery: int, smooth: float) -> None:
    apply_plot_style()
    fig, ax = new_axes(figsize=(3.35, 2.3))
    plotted = 0

    configs = sorted(expand_configs(patterns), key=lambda p: method_sort_key(_method_from_config(p)))
    for config_path in configs:
        method = _method_from_config(config_path)
        if method not in PLOT_METHODS:
            continue

        df = _load_mismatch(config_path)
        if df is None:
            print(f"[warn] no mismatch data for {config_path}")
            continue

        y = df["grad_mismatch_sq"].to_numpy()
        if smooth and smooth > 0:
            # smooth in log-space so periodic resets (orders-of-magnitude dips) don't
            # dominate the linear-space mean.
            logy = np.log(np.clip(y, 1e-30, None))
            y = np.exp(gaussian_smoothing(logy, sigma=smooth))
        ax.plot(
            df["round"].to_numpy(),
            y,
            color=method_color(method),
            label=method_label(method),
            lw=1,
            markevery=markevery,
        )
        plotted += 1

    if plotted == 0:
        print(f"[skip] {name}: no readable mismatch logs")
        plt.close(fig)
        return

    ax.set_xlabel("Number of rounds")
    ax.set_ylabel("Gradient Mismatch")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(frameon=False)
    save_and_close(fig, out_dir, f"gradient_mismatch_{name}.pdf")


def main() -> None:
    args = parse_args()
    _, manifest, out_dir = resolve_paths(args)
    entry = manifest.get("figures", {}).get("fig_mismatch", {})
    datasets = entry.get("datasets", {
        "cifar10": ["configs/gradient_mismatch/cifar10/*.toml"],
        "cifar100": ["configs/gradient_mismatch/cifar100/*.toml"],
    })
    for dataset_name, patterns in datasets.items():
        plot_dataset(dataset_name, patterns, out_dir, args.markevery, args.smooth)


if __name__ == "__main__":
    main()

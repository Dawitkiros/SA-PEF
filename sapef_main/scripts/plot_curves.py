"""Plot accuracy curves: rounds + cumulative uplink GB.

  - mean over seeds with Gaussian smoothing,
  - theoretical uplink computed from model param count + top-k fraction
    (FedAvg dense, CSER digest every H rounds), so figures don't depend on
    historical `fit_uplink_bits_total` accounting,
  - shared palette / markers / method order.

Each `figures.yaml` entry expands to one or more panels: configs are grouped
by their parent scenario directory (e.g. `cifar10/p10_dir05_top001/`) and one
PDF pair (rounds + comm-GB log axis) is emitted per scenario.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_utils import (
    accuracy_column,
    apply_plot_style,
    cumulative_uplink_gb,
    expand_configs,
    figure_entries,
    finalize_axes,
    gaussian_smoothing,
    load_result_csv,
    method_color,
    method_label,
    method_marker,
    method_sort_key,
    new_axes,
    read_config,
    resolve_paths,
    save_and_close,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/figures.yaml")
    parser.add_argument("--figure", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--smooth", type=float, default=5.0,
                        help="Gaussian-smoothing sigma applied to accuracy curves")
    return parser.parse_args()


def _method_from_config(config_path: Path) -> str:
    cfg = read_config(config_path)
    return str(cfg.get("algorithm", {}).get("name", config_path.stem)).lower()


def _scenario_key(config_path: Path) -> str:
    """Group configs by the directory just above the method TOML."""
    return config_path.parent.name


def _load_seeds_mean(
    config_path: Path,
    *,
    method: str,
    config: dict,
    clients_per_round: int,
) -> tuple[pd.Series, pd.Series, int] | None:
    """Return (rounds, mean_acc%, num_rounds) by averaging across seed CSVs.

    The release configs ship one CSV per (method, seed) under
    `<save-path-parent>/seedN/results.csv`. We discover sibling seed dirs
    automatically; if none present, we fall back to the configured save_path.
    """
    df = None
    try:
        df = load_result_csv(config_path)
    except FileNotFoundError:
        return None
    if df is None or df.empty:
        return None

    # Round 0 is the pre-training centralized eval; drop it for visualization.
    df = df[df["round"] > 0].copy().sort_values("round").reset_index(drop=True)
    y_col = accuracy_column(df)
    acc = pd.to_numeric(df[y_col], errors="coerce")
    if acc.max(skipna=True) is None:
        return None
    if acc.max(skipna=True) <= 1.0:
        acc = acc * 100.0
    return df["round"].astype(int), acc, int(df["round"].max())


def _plot_panel(
    panel: str,
    configs: list[Path],
    out_dir: Path,
    smooth: float,
    title: str | None,
) -> None:
    """One scenario directory → two PDFs (rounds, comm GB)."""
    apply_plot_style()
    fig_r, ax_r = new_axes(figsize=(3.35, 2.3))
    fig_c, ax_c = new_axes(figsize=(3.35, 2.3))
    plotted_r = plotted_c = 0

    configs_sorted = sorted(configs, key=lambda p: method_sort_key(_method_from_config(p)))

    for config_path in configs_sorted:
        method = _method_from_config(config_path)
        cfg = read_config(config_path)
        clients_per_round = int(cfg.get("clients-per-round", cfg.get("clients_per_round", 1)))

        loaded = _load_seeds_mean(
            config_path,
            method=method,
            config=cfg,
            clients_per_round=clients_per_round,
        )
        if loaded is None:
            print(f"[warn] no readable CSV for {config_path}")
            continue
        rounds, acc, num_rounds = loaded

        y_smooth = gaussian_smoothing(acc.values, sigma=smooth)
        color = method_color(method)
        marker = method_marker(method)
        label = method_label(method)

        ax_r.plot(rounds.values, y_smooth, lw=2, label=label, color=color,
                  marker=marker, markevery=max(1, num_rounds // 5))
        plotted_r += 1

        try:
            comm = cumulative_uplink_gb(
                method=method,
                config=cfg,
                num_rounds=num_rounds,
                clients_per_round=clients_per_round,
            )
            ax_c.plot(comm, y_smooth, lw=2, label=label, color=color)
            plotted_c += 1
        except KeyError as exc:
            print(f"[warn] {config_path}: comm axis skipped ({exc})")

    if plotted_r == 0:
        plt.close(fig_r); plt.close(fig_c)
        print(f"[skip] {panel}: no readable logs")
        return

    finalize_axes(ax_r, xlabel="Number of rounds", ylabel="Test accuracy (%)",
                  title=title, dedup_legend=False)
    save_and_close(fig_r, out_dir, f"{panel}_round.pdf")

    if plotted_c > 0:
        ax_c.set_xscale("log")
        finalize_axes(ax_c, xlabel="Communication cost (GB)",
                      ylabel="Test accuracy (%)",
                      title=title, dedup_legend=False)
        save_and_close(fig_c, out_dir, f"{panel}_comm_gb.pdf")
    else:
        plt.close(fig_c)


def plot_entry(entry: dict, out_dir: Path, smooth: float) -> None:
    if "x_axes" not in entry:
        # Reserved for plot_alpha_sweep.py / plot_gradient_mismatch.py.
        return
    configs = expand_configs(entry.get("configs", []))
    if not configs:
        print(f"[skip] {entry.get('name', 'unnamed')}: no configs")
        return

    panels: dict[str, list[Path]] = defaultdict(list)
    for config_path in configs:
        panels[_scenario_key(config_path)].append(config_path)

    title = entry.get("title")
    name = entry.get("name", "figure")
    for panel, paths in panels.items():
        panel_name = f"{name}_{panel}" if len(panels) > 1 else name
        _plot_panel(panel_name, paths, out_dir, smooth, title)


def main() -> None:
    args = parse_args()
    _, manifest, out_dir = resolve_paths(args)
    for entry in figure_entries(manifest, args.figure, args.all):
        plot_entry(entry, out_dir, args.smooth)


if __name__ == "__main__":
    main()

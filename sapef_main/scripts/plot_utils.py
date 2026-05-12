"""Shared plotting helpers."""
from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:  # graceful fallback if scipy missing
    def gaussian_filter1d(arr, sigma):
        return np.asarray(arr, dtype=float)


METHOD_NAMES = {
    "fedavg":   "FedAvg",
    "ef":       "EF",
    "saef":     "SAEF",
    "cser":     "CSER",
    "sapef":    "SA-PEF",
    "scaffold": "SCAFFOLD",
    "scallion": "SCALLION",
    "scafcom":  "SCAFCOM",
}

# Palette / markers / method order for accuracy curves.
PALETTE = {
    "fedavg":   "#ff7f00",  # orange
    "ef":       "#377eb8",  # blue
    "saef":     "#e41a1c",  # red
    "cser":     "#984ea3",  # purple
    "sapef":    "#4daf4a",  # green
    "scaffold": "#a65628",  # brown
    "scallion": "#f781bf",  # pink
    "scafcom":  "#999999",  # grey
}
MARKERS = {
    "fedavg":   "^",
    "ef":       "s",
    "saef":     "o",
    "cser":     "P",
    "sapef":    "D",
    "scaffold": "v",
    "scallion": "X",
    "scafcom":  "*",
}
METHOD_ORDER = [
    "sapef", "ef", "saef", "cser", "fedavg",
    "scafcom", "scaffold", "scallion",
]

# Trainable-parameter counts for theoretical uplink accounting.
# Add to this map as new models are introduced.
MODEL_PARAMS = {
    "FEMNIST_CNN": 6_603_710,
    "ResNet9":     6_573_120,
    "ResNet18":   11_220_132,
    "ResNet34":   21_336_004,
}

DTYPE_BYTES = 4   # float32 values
INDEX_BYTES = 4   # int32 indices

# Per-alpha style for the SA-PEF ablation plot.
ALPHA_STYLE = {
    0.0: {"color": "#377eb8", "marker": "*", "label_prefix": "EF"},
    0.1: {"color": "#6F9eb8", "marker": "p", "label_prefix": "SA-PEF"},
    0.2: {"color": "#235280", "marker": "d", "label_prefix": "SA-PEF"},
    0.3: {"color": "#4A5662", "marker": "h", "label_prefix": "SA-PEF"},
    0.4: {"color": "#778CA2", "marker": "X", "label_prefix": "SA-PEF"},
    0.5: {"color": "#97B3D0", "marker": "v", "label_prefix": "SA-PEF"},
    0.6: {"color": "#00A300", "marker": "^", "label_prefix": "SA-PEF"},
    0.7: {"color": "#40826D", "marker": "P", "label_prefix": "SA-PEF"},
    0.8: {"color": "#004700", "marker": "s", "label_prefix": "SA-PEF"},
    0.9: {"color": "#008000", "marker": "o", "label_prefix": "SA-PEF"},
    1.0: {"color": "#e41a1c", "marker": "D", "label_prefix": "SAEF"},
}


def repo_root() -> Path:
    # scripts/<file>.py → repo (sapef_main/)
    return Path(__file__).resolve().parents[1]


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return data


def read_config(path: Path) -> dict:
    with path.open("rb") as fp:
        return tomllib.load(fp)


def config_save_path(config_path: Path) -> Path:
    cfg = read_config(config_path)
    raw = cfg.get("save-path") or cfg.get("save_path")
    if raw is None:
        raise KeyError(f"Config has no save-path: {config_path}")
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root() / path


def method_from_config(config_path: Path) -> str:
    cfg = read_config(config_path)
    method = cfg.get("algorithm", {}).get("name", config_path.stem)
    return METHOD_NAMES.get(str(method).lower(), str(method))


def load_result_csv(config_path: Path) -> pd.DataFrame:
    csv_path = config_save_path(config_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing results CSV for {config_path}: expected {csv_path}"
        )
    df = pd.read_csv(csv_path)
    if "round" not in df:
        raise ValueError(f"Results CSV has no 'round' column: {csv_path}")
    return df


def accuracy_column(df: pd.DataFrame) -> str:
    for col in ("centralized_accuracy", "distributed_accuracy", "accuracy"):
        if col in df.columns:
            return col
    candidates = [c for c in df.columns if c.endswith("_accuracy")]
    if candidates:
        return candidates[0]
    raise ValueError("No accuracy column found in results CSV")


def figure_entries(manifest: dict, figure: str | None, all_figures: bool) -> Iterable[dict]:
    figures = manifest.get("figures", {})
    if all_figures:
        return figures.values()
    if figure is None:
        raise ValueError("Pass --figure NAME or --all")
    if figure not in figures:
        raise KeyError(f"Unknown figure {figure!r}. Available: {', '.join(figures)}")
    return [figures[figure]]


def expand_configs(patterns: Iterable[str]) -> list[Path]:
    root = repo_root()
    out: list[Path] = []
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if not matches:
            print(f"[warn] no configs matched {pattern}", file=sys.stderr)
        out.extend(matches)
    return out


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, dict, Path]:
    """Resolve repo-relative `manifest` and `out_dir` from parsed args."""
    root = repo_root()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = load_manifest(manifest_path)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    return root, manifest, out_dir


def new_axes(figsize: Tuple[float, float] = (7.0, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def finalize_axes(
    ax,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    *,
    dedup_legend: bool = True,
) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if dedup_legend:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False)
    elif handles:
        ax.legend(frameon=False)


def save_and_close(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out}")
    return out


def apply_plot_style() -> None:
    """Apply the publication-quality matplotlib style."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rc("mathtext", fontset="cm")
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })


def gaussian_smoothing(y, sigma: float = 5.0) -> np.ndarray:
    return gaussian_filter1d(np.asarray(y, dtype=float), sigma=sigma)


def model_param_count(config: dict) -> int:
    """Trainable param count from the config's [model] block."""
    name = config.get("model", {}).get("name") or config.get("model_name")
    if name in MODEL_PARAMS:
        return MODEL_PARAMS[name]
    raise KeyError(
        f"Unknown model {name!r}; add it to MODEL_PARAMS in plot_utils.py"
    )


def uplink_bytes_fedavg(param_count: int) -> int:
    """FedAvg per-client per-round uplink: dense float32 model."""
    return param_count * DTYPE_BYTES


def uplink_bytes_topk(param_count: int, k_frac: float) -> int:
    """Top-k per-client per-round uplink: k float32 values + k int32 indices."""
    k = int(math.ceil(param_count * k_frac))
    return k * (DTYPE_BYTES + INDEX_BYTES)


def uplink_bytes_cser(
    param_count: int,
    k_frac: float,
    round_idx: int,
    H: int,
    reset_frac: float,
) -> int:
    """CSER per-client per-round uplink: top-k each round + digest every H rounds."""
    base = uplink_bytes_topk(param_count, k_frac)
    if H > 0 and (round_idx % H == 0):
        k_extra = int(math.ceil(param_count * reset_frac))
        base += k_extra * (DTYPE_BYTES + INDEX_BYTES)
    return base


def cumulative_uplink_gb(
    method: str,
    config: dict,
    num_rounds: int,
    clients_per_round: int,
) -> np.ndarray:
    """Cumulative uplink (GB) over num_rounds, summed across selected clients per round."""
    params = model_param_count(config)
    method_lc = method.lower()
    k_frac = float(config.get("sparsify-by", config.get("sparsify_by", 0.0)))

    H = int(config.get("H", 5))
    reset_frac = float(config.get("reset-frac", config.get("reset_frac", 0.1)))

    cum = np.zeros(num_rounds, dtype=np.float64)
    running = 0.0
    for r in range(1, num_rounds + 1):
        if method_lc == "fedavg":
            ub = uplink_bytes_fedavg(params)
        elif method_lc == "cser":
            ub = uplink_bytes_cser(params, k_frac, r, H, reset_frac)
        else:
            ub = uplink_bytes_topk(params, k_frac)
        running += ub * clients_per_round
        cum[r - 1] = running / 1e9
    return cum


def method_color(method: str) -> str | None:
    return PALETTE.get(method.lower())


def method_marker(method: str) -> str | None:
    return MARKERS.get(method.lower())


def method_label(method: str) -> str:
    return METHOD_NAMES.get(method.lower(), method)


def method_sort_key(method: str) -> int:
    m = method.lower()
    return METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER)

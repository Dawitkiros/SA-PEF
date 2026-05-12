"""Plot actual class distributions for pathological-1 and pathological-2 partitions.

Uses the same McMahan-style shard partitioning as dataset.py so the plots
match what the training runs see.  Generates per-partition:
  1. Heatmap of class proportions per client (sorted by dominant class)
  2. Histogram of # classes per client + # samples per client
  3. Side-by-side comparison: natural vs patho1 vs patho2 vs Dirichlet(0.1)

Usage:
    HF_HOME=/path/to/hf_cache python scripts/femnist_plots/plot_patho_distribution.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

NUM_CLIENTS = 200
SEED = 42
NUM_CLASSES = 62
OUT_DIR = "plots"

hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/hf_cache"))
os.environ.setdefault("HF_HOME", hf_home)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
os.makedirs(OUT_DIR, exist_ok=True)


# Partitioning functions (mirrored from dataset.py).

def pathological_partition(labels, num_clients, num_classes_per_client, rng):
    """McMahan-style: sort by label, split into shards, assign round-robin."""
    total_shards = num_clients * num_classes_per_client
    sorted_idx = np.argsort(labels, kind="stable")
    shards = np.array_split(sorted_idx, total_shards)
    shard_order = np.arange(total_shards)
    rng.shuffle(shard_order)
    partitions = []
    for c in range(num_clients):
        client_idx = []
        for s in range(num_classes_per_client):
            shard_id = shard_order[c * num_classes_per_client + s]
            client_idx.extend(shards[shard_id].tolist())
        partitions.append(client_idx)
    return partitions


def dirichlet_partition(labels, num_clients, alpha, rng):
    """Dirichlet(alpha) partition over classes."""
    num_classes = int(labels.max()) + 1
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    for idx_arr in class_indices:
        rng.shuffle(idx_arr)
    partitions = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx_c = class_indices[c]
        if len(idx_c) == 0:
            continue
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        counts = (proportions * len(idx_c)).astype(int)
        remainder = len(idx_c) - counts.sum()
        fracs = (proportions * len(idx_c)) - counts
        top_clients = np.argsort(fracs)[-remainder:]
        counts[top_clients] += 1
        offset = 0
        for k in range(num_clients):
            partitions[k].extend(idx_c[offset : offset + counts[k]].tolist())
            offset += counts[k]
    return partitions


def compute_client_stats(partitions, all_labels):
    """Compute per-client class count matrix and summary stats."""
    n = len(partitions)
    cc = np.zeros((n, NUM_CLASSES), dtype=int)
    num_samples = []
    num_classes = []
    for cid, indices in enumerate(partitions):
        labels = all_labels[indices]
        num_samples.append(len(indices))
        unique = set(labels.tolist())
        num_classes.append(len(unique))
        for lbl in labels:
            cc[cid, lbl] += 1
    return cc, np.array(num_samples), np.array(num_classes)


def print_stats(name, num_samples, num_classes):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Total samples: {num_samples.sum()}")
    print(f"  Samples/client:  min={num_samples.min()}, "
          f"median={int(np.median(num_samples))}, "
          f"mean={num_samples.mean():.1f}, max={num_samples.max()}")
    print(f"  Classes/client:  min={num_classes.min()}, "
          f"median={int(np.median(num_classes))}, "
          f"mean={num_classes.mean():.1f}, max={num_classes.max()}")

    log_base = np.log(NUM_CLASSES)
    ents = []
    for cid in range(len(num_samples)):
        total = num_samples[cid]
        if total == 0:
            ents.append(0.0)
            continue
        # recompute from num_classes won't work, need cc — skip here
    print()


def plot_heatmap(cc, num_classes_arr, title, fname):
    sort_idx = np.argsort(num_classes_arr)
    sorted_counts = cc[sort_idx]
    row_sums = sorted_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    sorted_props = sorted_counts / row_sums

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(sorted_props, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_xlabel("Class (character 0-61)", fontsize=12)
    ax.set_ylabel("Client (sorted by # classes)", fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.colorbar(im, ax=ax, label="Proportion of client's data")
    plt.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"Saved: {fname}")
    plt.close(fig)


def plot_histograms(num_classes_arr, num_samples, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.hist(num_classes_arr, bins=range(0, NUM_CLASSES + 2),
             edgecolor="black", alpha=0.7, color="steelblue")
    ax1.set_xlabel("Number of classes per client", fontsize=12)
    ax1.set_ylabel("Number of clients", fontsize=12)
    ax1.set_title(f"{title}: Class Coverage", fontsize=13)
    ax1.axvline(num_classes_arr.mean(), color="red", linestyle="--",
                label=f"Mean = {num_classes_arr.mean():.1f}")
    ax1.axvline(np.median(num_classes_arr), color="orange", linestyle="--",
                label=f"Median = {int(np.median(num_classes_arr))}")
    ax1.legend()

    ax2 = axes[1]
    ax2.hist(num_samples, bins=30, edgecolor="black", alpha=0.7, color="coral")
    ax2.set_xlabel("Number of samples per client", fontsize=12)
    ax2.set_ylabel("Number of clients", fontsize=12)
    ax2.set_title(f"{title}: Sample Count", fontsize=13)
    ax2.axvline(num_samples.mean(), color="red", linestyle="--",
                label=f"Mean = {num_samples.mean():.0f}")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"Saved: {fname}")
    plt.close(fig)


print("[plot] Loading flwrlabs/femnist ...")
full = load_dataset("flwrlabs/femnist", split="train")
all_labels = np.array(full["character"])
writer_col = full["writer_id"]
print(f"  Total samples: {len(all_labels)}, classes: {len(set(all_labels.tolist()))}")

print("\n[plot] Natural partition ...")
unique_writers = sorted(set(writer_col))
rng_nat = np.random.default_rng(SEED)
rng_nat.shuffle(unique_writers)
selected = unique_writers[:NUM_CLIENTS]
selected_set = set(selected)
by_writer = {}
for i, w in enumerate(writer_col):
    if w in selected_set:
        by_writer.setdefault(w, []).append(i)
nat_partitions = [by_writer.get(w, []) for w in selected]
nat_cc, nat_samples, nat_classes = compute_client_stats(nat_partitions, all_labels)
print_stats("Natural (200 writers)", nat_samples, nat_classes)

print("[plot] Pathological-1 partition ...")
rng_p1 = np.random.default_rng(SEED)
p1_parts = pathological_partition(all_labels, NUM_CLIENTS, 1, rng_p1)
p1_cc, p1_samples, p1_classes = compute_client_stats(p1_parts, all_labels)
print_stats("Pathological-1 (1 shard/client)", p1_samples, p1_classes)

print("[plot] Pathological-2 partition ...")
rng_p2 = np.random.default_rng(SEED)
p2_parts = pathological_partition(all_labels, NUM_CLIENTS, 2, rng_p2)
p2_cc, p2_samples, p2_classes = compute_client_stats(p2_parts, all_labels)
print_stats("Pathological-2 (2 shards/client)", p2_samples, p2_classes)

print("[plot] Dirichlet(alpha=0.1) partition ...")
rng_dir = np.random.default_rng(SEED)
dir_parts = dirichlet_partition(all_labels, NUM_CLIENTS, 0.1, rng_dir)
dir_cc, dir_samples, dir_classes = compute_client_stats(dir_parts, all_labels)
print_stats("Dirichlet(0.1)", dir_samples, dir_classes)

plot_heatmap(p1_cc, p1_classes,
             f"Pathological-1: Per-Client Class Distribution\n"
             f"(200 clients, {p1_classes.mean():.1f} classes/client, "
             f"median {int(np.median(p1_classes))})",
             f"{OUT_DIR}/femnist_patho1_heatmap.png")

plot_heatmap(p2_cc, p2_classes,
             f"Pathological-2: Per-Client Class Distribution\n"
             f"(200 clients, {p2_classes.mean():.1f} classes/client, "
             f"median {int(np.median(p2_classes))})",
             f"{OUT_DIR}/femnist_patho2_heatmap.png")

plot_heatmap(dir_cc, dir_classes,
             f"Dirichlet(0.1): Per-Client Class Distribution\n"
             f"(200 clients, {dir_classes.mean():.1f} classes/client, "
             f"median {int(np.median(dir_classes))})",
             f"{OUT_DIR}/femnist_dir01_heatmap.png")

plot_histograms(p1_classes, p1_samples, "Pathological-1",
                f"{OUT_DIR}/femnist_patho1_histograms.png")
plot_histograms(p2_classes, p2_samples, "Pathological-2",
                f"{OUT_DIR}/femnist_patho2_histograms.png")
plot_histograms(dir_classes, dir_samples, "Dirichlet(0.1)",
                f"{OUT_DIR}/femnist_dir01_histograms.png")

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

configs = [
    (nat_cc, nat_classes, f"Natural\n(mean {nat_classes.mean():.0f} cls, "
     f"median {int(np.median(nat_classes))})"),
    (p1_cc, p1_classes, f"Pathological-1\n(mean {p1_classes.mean():.1f} cls, "
     f"median {int(np.median(p1_classes))})"),
    (p2_cc, p2_classes, f"Pathological-2\n(mean {p2_classes.mean():.1f} cls, "
     f"median {int(np.median(p2_classes))})"),
    (dir_cc, dir_classes, f"Dirichlet(0.1)\n(mean {dir_classes.mean():.1f} cls, "
     f"median {int(np.median(dir_classes))})"),
]

for ax, (cc, nc, title) in zip(axes.flat, configs):
    sort_idx = np.argsort(nc)
    sorted_c = cc[sort_idx]
    rs = sorted_c.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    props = sorted_c / rs
    im = ax.imshow(props, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Class (0-61)")
    ax.set_ylabel("Client (sorted)")
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("FEMNIST: Per-Client Class Distribution Across Partitioning Strategies",
             fontsize=15, y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/femnist_all_partitions_heatmaps.png", dpi=150,
            bbox_inches="tight")
print(f"\nSaved: {OUT_DIR}/femnist_all_partitions_heatmaps.png")

fig2, axes2 = plt.subplots(1, 4, figsize=(22, 5))
hist_configs = [
    (nat_classes, "Natural", "steelblue"),
    (p1_classes, "Pathological-1", "tomato"),
    (p2_classes, "Pathological-2", "orangered"),
    (dir_classes, "Dirichlet(0.1)", "mediumpurple"),
]
for ax, (nc, title, color) in zip(axes2, hist_configs):
    ax.hist(nc, bins=range(0, NUM_CLASSES + 2), edgecolor="black", alpha=0.7,
            color=color)
    ax.set_xlim(0, NUM_CLASSES + 1)
    ax.set_xlabel("# classes per client")
    ax.set_ylabel("# clients")
    ax.set_title(f"{title}\n(mean={nc.mean():.1f}, med={int(np.median(nc))})",
                 fontsize=11)

plt.suptitle("FEMNIST: Class Coverage Distribution per Partitioning Strategy",
             fontsize=14, y=1.02)
plt.tight_layout()
fig2.savefig(f"{OUT_DIR}/femnist_all_partitions_histograms.png", dpi=150,
             bbox_inches="tight")
print(f"Saved: {OUT_DIR}/femnist_all_partitions_histograms.png")

print("\nDone.")

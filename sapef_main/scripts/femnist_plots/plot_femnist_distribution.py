"""Analyze and plot the FEMNIST natural partitioning distribution.

Shows per-client class coverage to quantify heterogeneity under
NaturalIdPartitioner (writer_id). Generates:
  1. Heatmap of class-count per client (sorted by # classes)
  2. Histogram of # classes per client
  3. Summary stats: mean/median/min/max classes per client, Gini index

Usage:
    HF_HOME=/path/to/hf_cache python scripts/femnist_plots/plot_femnist_distribution.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

NUM_CLIENTS = 200
SEED = 42
OUT_DIR = "plots"

hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/hf_cache"))
os.environ.setdefault("HF_HOME", hf_home)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))

os.makedirs(OUT_DIR, exist_ok=True)

# Load + partition (mirrors dataset.py logic).
print("[plot] Loading flwrlabs/femnist ...")
full = load_dataset("flwrlabs/femnist", split="train")
writer_col = full["writer_id"]
char_col = np.array(full["character"])

unique_writers = sorted(set(writer_col))
rng = np.random.default_rng(SEED)
rng.shuffle(unique_writers)
selected = unique_writers[:NUM_CLIENTS]
selected_set = set(selected)

by_writer = {}
for i, w in enumerate(writer_col):
    if w in selected_set:
        by_writer.setdefault(w, []).append(i)

NUM_CLASSES = 62

client_class_counts = np.zeros((NUM_CLIENTS, NUM_CLASSES), dtype=int)
client_num_samples = []
client_num_classes = []

for cid, w in enumerate(selected):
    indices = by_writer.get(w, [])
    labels = char_col[indices]
    client_num_samples.append(len(indices))
    unique_labels = set(labels.tolist())
    client_num_classes.append(len(unique_labels))
    for lbl in labels:
        client_class_counts[cid, lbl] += 1

client_num_samples = np.array(client_num_samples)
client_num_classes = np.array(client_num_classes)

print(f"\n{'='*60}")
print(f"FEMNIST Natural Partition Analysis (200 writers, seed={SEED})")
print(f"{'='*60}")
print(f"Total samples across 200 clients: {client_num_samples.sum()}")
print(f"Samples per client:  min={client_num_samples.min()}, "
      f"median={int(np.median(client_num_samples))}, "
      f"mean={client_num_samples.mean():.1f}, "
      f"max={client_num_samples.max()}")
print(f"Classes per client:  min={client_num_classes.min()}, "
      f"median={int(np.median(client_num_classes))}, "
      f"mean={client_num_classes.mean():.1f}, "
      f"max={client_num_classes.max()}")

log_base = np.log(NUM_CLASSES)
entropies = []
for cid in range(NUM_CLIENTS):
    counts = client_class_counts[cid]
    total = counts.sum()
    if total == 0:
        entropies.append(0.0)
        continue
    probs = counts[counts > 0] / total
    ent = -np.sum(probs * np.log(probs)) / log_base
    entropies.append(ent)
entropies = np.array(entropies)
print(f"Normalized entropy:  min={entropies.min():.3f}, "
      f"median={np.median(entropies):.3f}, "
      f"mean={entropies.mean():.3f}, "
      f"max={entropies.max():.3f}")
print(f"  (1.0 = uniform over 62 classes, 0.0 = single class)")

global_coverage = (client_class_counts.sum(axis=0) > 0).sum()
print(f"Global class coverage: {global_coverage}/62")

sort_idx = np.argsort(client_num_classes)
sorted_counts = client_class_counts[sort_idx]
row_sums = sorted_counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
sorted_props = sorted_counts / row_sums

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(sorted_props, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax.set_xlabel("Class (character 0–61)", fontsize=12)
ax.set_ylabel("Client (sorted by # classes)", fontsize=12)
ax.set_title(f"FEMNIST Natural Partition: Per-Client Class Distribution\n"
             f"(200 writers, mean {client_num_classes.mean():.0f} classes/client, "
             f"median {int(np.median(client_num_classes))})", fontsize=13)
plt.colorbar(im, ax=ax, label="Proportion of client's data")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/femnist_natural_heatmap.png", dpi=150)
print(f"\nSaved: {OUT_DIR}/femnist_natural_heatmap.png")

fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.hist(client_num_classes, bins=range(0, NUM_CLASSES + 2), edgecolor="black",
         alpha=0.7, color="steelblue")
ax1.set_xlabel("Number of classes per client", fontsize=12)
ax1.set_ylabel("Number of clients", fontsize=12)
ax1.set_title("Class Coverage Distribution", fontsize=13)
ax1.axvline(client_num_classes.mean(), color="red", linestyle="--",
            label=f"Mean = {client_num_classes.mean():.1f}")
ax1.axvline(np.median(client_num_classes), color="orange", linestyle="--",
            label=f"Median = {int(np.median(client_num_classes))}")
ax1.legend()

ax2 = axes[1]
ax2.hist(client_num_samples, bins=30, edgecolor="black", alpha=0.7, color="coral")
ax2.set_xlabel("Number of samples per client", fontsize=12)
ax2.set_ylabel("Number of clients", fontsize=12)
ax2.set_title("Sample Count Distribution", fontsize=13)
ax2.axvline(client_num_samples.mean(), color="red", linestyle="--",
            label=f"Mean = {client_num_samples.mean():.0f}")
ax2.legend()

plt.tight_layout()
fig2.savefig(f"{OUT_DIR}/femnist_natural_histograms.png", dpi=150)
print(f"Saved: {OUT_DIR}/femnist_natural_histograms.png")

# Comparison with pathological + Dirichlet(0.1) (simulated counts).
fig3, axes3 = plt.subplots(1, 4, figsize=(20, 5))

titles = [
    f"Natural (mean {client_num_classes.mean():.0f} cls)",
    "Pathological 1-class",
    "Pathological 2-class",
    "Dirichlet(α=0.1)"
]

axes3[0].hist(client_num_classes, bins=range(0, NUM_CLASSES + 2),
              edgecolor="black", alpha=0.7, color="steelblue")
axes3[0].set_xlim(0, NUM_CLASSES + 1)

patho1 = np.ones(NUM_CLIENTS, dtype=int)
axes3[1].hist(patho1, bins=range(0, NUM_CLASSES + 2),
              edgecolor="black", alpha=0.7, color="tomato")
axes3[1].set_xlim(0, NUM_CLASSES + 1)

patho2 = 2 * np.ones(NUM_CLIENTS, dtype=int)
axes3[2].hist(patho2, bins=range(0, NUM_CLASSES + 2),
              edgecolor="black", alpha=0.7, color="tomato")
axes3[2].set_xlim(0, NUM_CLASSES + 1)

# Sample class counts from Dirichlet(0.1) for the rightmost panel.
dir_rng = np.random.default_rng(42)
n_per_client = int(client_num_samples.mean())
dir_classes_per_client = []
for _ in range(NUM_CLIENTS):
    probs = dir_rng.dirichlet(0.1 * np.ones(NUM_CLASSES))
    counts = dir_rng.multinomial(n_per_client, probs)
    dir_classes_per_client.append((counts > 0).sum())
axes3[3].hist(dir_classes_per_client, bins=range(0, NUM_CLASSES + 2),
              edgecolor="black", alpha=0.7, color="mediumpurple")
axes3[3].set_xlim(0, NUM_CLASSES + 1)

for i, ax in enumerate(axes3):
    ax.set_xlabel("# classes per client")
    ax.set_ylabel("# clients")
    ax.set_title(titles[i], fontsize=11)

plt.suptitle("FEMNIST: Heterogeneity Comparison Across Partitioning Strategies",
             fontsize=14, y=1.02)
plt.tight_layout()
fig3.savefig(f"{OUT_DIR}/femnist_partition_comparison.png", dpi=150,
             bbox_inches="tight")
print(f"Saved: {OUT_DIR}/femnist_partition_comparison.png")

print("\nDone.")

"""One-shot HF download of flwrlabs/femnist.

Run this once on a host that shares the cache filesystem with your training
nodes to warm the HF cache that subsequent jobs will then consume:

    HF_HOME=/path/to/hf_cache python scripts/prefetch_femnist.py

After this runs, training jobs hit the cached copy instead of re-downloading
on every job.
"""
import os
import time

from datasets import load_dataset

hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/hf_cache"))
os.environ.setdefault("HF_HOME", hf_home)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

print(f"[prefetch] HF_HOME={hf_home}")
print(f"[prefetch] HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}")
print("[prefetch] downloading flwrlabs/femnist train split ...")
t0 = time.time()
ds = load_dataset("flwrlabs/femnist", split="train")
print(f"[prefetch] done in {time.time() - t0:.1f}s; {len(ds)} examples")
unique_writers = len(set(ds["writer_id"]))
print(f"[prefetch] {unique_writers} unique writers")

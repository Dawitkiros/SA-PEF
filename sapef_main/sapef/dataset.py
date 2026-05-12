"""Dataset loading and partitioning."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, List
from datasets import DatasetDict, load_dataset
from easydict import EasyDict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DistributionPartitioner
from flwr_datasets.preprocessor import Preprocessor
from torchvision.transforms import Compose, Normalize, ToTensor


FDS = None  # Cache FederatedDataset

MNIST_TRANSFORMS = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])


def get_transforms(dataset_name: str):
    """Get transforms for dataset."""
    if dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif dataset_name == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    elif dataset_name == "mnist":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return transform_train, transform_test


def partition_dirichlet(
    dataset,
    num_clients: int,
    alpha: float,
    seed: int = 42
) -> List[List[int]]:
    """Partition dataset using Dirichlet distribution."""
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))

    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        rng.shuffle(idx_k)

        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

        idx_k_split = np.split(idx_k, proportions)
        for i, idx in enumerate(idx_k_split):
            client_indices[i].extend(idx.tolist())

    return client_indices


from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch

def partition_mnist_shards(
    trainset,
    num_clients: int,
    num_shards: int = 400,
    shards_per_client: int = 2,
    seed: int = 42,
):
    """SCALLION-style partition: 400 single-class shards, 2 per client."""
    assert num_shards == num_clients * shards_per_client, \
        "num_shards must equal num_clients * shards_per_client"

    rng = np.random.RandomState(seed)

    targets = np.array(trainset.targets if hasattr(trainset, "targets") else trainset.labels)
    num_classes = len(np.unique(targets))
    shards_per_class = num_shards // num_classes

    shards = []
    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        shard_size = len(idx_c) // shards_per_class
        for s in range(shards_per_class):
            start = s * shard_size
            end = (s + 1) * shard_size if s < shards_per_class - 1 else len(idx_c)
            shards.append(idx_c[start:end])

    rng.shuffle(shards)

    client_indices = []
    for cid in range(num_clients):
        s_start = cid * shards_per_client
        s_end = s_start + shards_per_client
        client_shards = shards[s_start:s_end]
        idx = np.concatenate(client_shards).astype(int)
        rng.shuffle(idx)
        client_indices.append(idx.tolist())

    return client_indices


def load_partition_data(partition_id: int, config: dict) -> Tuple[DataLoader, DataLoader]:
    """Load data for a specific partition (CIFAR + MNIST/F-MNIST + FEMNIST)."""
    dataset_name = config["dataset"]["name"].lower()
    num_clients = config["num_clients"]
    batch_size = config["dataset"]["batch_size"]
    alpha = config.get("dataset", {}).get("alpha", 0.5)
    seed = config.get("dataset", {}).get("seed", 42)

    # FEMNIST uses a naturally-partitioned (per-writer) pipeline via HF + flwr_datasets.
    if dataset_name == "femnist":
        return _femnist_partition_loaders(partition_id, config)

    transform_train, _ = get_transforms(dataset_name)

    if dataset_name == "cifar10":
        trainset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
    elif dataset_name == "cifar100":
        trainset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
    elif dataset_name == "mnist":
        trainset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_train
        )
    elif dataset_name in ["fashion-mnist", "fmnist"]:
        trainset = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform_train
        )
        dataset_name = "fashion-mnist"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    partitioning = config["dataset"].get("partitioning", "dirichlet")

    if partitioning == "iid":
        total_samples = len(trainset)
        samples_per_client = total_samples // num_clients
        start_idx = partition_id * samples_per_client
        end_idx = start_idx + samples_per_client if partition_id < num_clients - 1 else total_samples
        indices = list(range(start_idx, end_idx))

    elif partitioning == "dirichlet":
        client_indices = partition_dirichlet(trainset, num_clients, alpha, seed)
        indices = client_indices[partition_id]

    elif partitioning in ["shards", "scallion"]:
        if dataset_name not in ["mnist", "fashion-mnist"]:
            raise ValueError("Shards partitioning is only intended for (F)MNIST here.")
        client_indices = partition_mnist_shards(
            trainset,
            num_clients=num_clients,
            num_shards=400,
            shards_per_client=2,
            seed=seed,
        )
        indices = client_indices[partition_id]

    else:
        raise ValueError(f"Unknown partitioning: {partitioning}")

    client_dataset = Subset(trainset, indices)

    val_split = 0.2
    n_val = int(len(client_dataset) * val_split)
    n_train = len(client_dataset) - n_val

    train_subset, val_subset = torch.utils.data.random_split(
        client_dataset, [n_train, n_val]
    )

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=batch_size)

    return trainloader, valloader

def load_test_data(run_config):
    dataset_name = run_config["dataset"]["name"].lower()

    # FEMNIST centralized test: pooled holdout across the sampled writers.
    if dataset_name == "femnist":
        return _femnist_test_loader(run_config)

    _, transform_test = get_transforms(dataset_name)
    batch_size_test = run_config["dataset"].get("batch_size", 256)

    if dataset_name == "cifar10":
        testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    elif dataset_name == "cifar100":
        testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    elif dataset_name == "mnist":
        testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)
    elif dataset_name in ["fashion-mnist", "fmnist"]:
        testset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(testset, batch_size=batch_size_test, shuffle=False)


class FEMNISTFilter(Preprocessor): #type: ignore
    """A Preprocessor class that filter the FEMNIST data.

    It filters data with label 0 to 9 (lower case letters 'a'-'j')
    """

    def __call__(self, dataset: DatasetDict) -> DatasetDict:
        """."""
        allowed_labels = list(range(10))  # mapping to 'a'-'j'
        filtered_dataset = dataset.filter(
            lambda example: example["character"] in allowed_labels
        )
        return filtered_dataset


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [MNIST_TRANSFORMS(img) for img in batch["image"]]

    return batch


def process_femnist(dataset):
    """Process FEMNIST when setting up centralised test data."""
    return dataset.filter(lambda example: example["character"] in list(range(10)))


def load_data(
    dataset_config: EasyDict,
    partition_id: int,
    num_partitions: int,
):
    """Load and partition data."""
    # Only initialize `FederatedDataset` once
    global FDS  # pylint: disable=global-statement
    if FDS is None:
        # Generate a vector from a log-normal probability distribution
        rng = np.random.default_rng(dataset_config.seed) #type: ignore
        distribution_array = rng.lognormal(
            dataset_config.mu, #type: ignore
            dataset_config.sigma, #type: ignore
            (num_partitions * dataset_config.num_unique_labels_per_partition), #type: ignore
        )
        distribution_array = distribution_array.reshape(
            (dataset_config.num_unique_labels, -1) #type: ignore
        )
        labels_per_partition = dataset_config.num_unique_labels_per_partition #type: ignore
        samples_per_label = dataset_config.preassigned_num_samples_per_label #type: ignore
        label_key = "character" if "femnist" in dataset_config.path else "label" #type: ignore
        partitioner = DistributionPartitioner(
            distribution_array=distribution_array,
            num_partitions=num_partitions,
            num_unique_labels_per_partition=labels_per_partition,
            partition_by=label_key,  # target column `label` ("character" for FEMNIST)
            preassigned_num_samples_per_label=samples_per_label,
        )
        if "femnist" in dataset_config.path: #type: ignore
            FDS = FederatedDataset(
                dataset=dataset_config.path, #type: ignore
                partitioners={"train": partitioner},
                preprocessor=FEMNISTFilter(),  # Add the Preprocessor class for FEMNIST
            )
        else:
            FDS = FederatedDataset(
                dataset=dataset_config.path, #type: ignore
                partitioners={"train": partitioner},
            )

    partition = FDS.load_partition(partition_id)

    # Divide data on each node: 90% train, 10% test
    partition_train_test = partition.train_test_split(
        test_size=dataset_config.val_ratio, seed=dataset_config.seed #type: ignore
    )
    # The validation set is never used because we do centralized evaluation
    # on the server on the held-out test dataset.
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    return (
        DataLoader(
            partition_train_test["train"], #type: ignore
            batch_size=dataset_config.batch_size, #type: ignore
            shuffle=True,
        ),
        DataLoader(
            partition_train_test["test"], #type: ignore
            batch_size=dataset_config.batch_size, #type: ignore
        ),
    )


def prepare_test_loader(dataset_config: EasyDict):
    """Generate the dataloader for the test set.

    Args:
        dataset_config (dict): The dataset configuration.

    Note: FEMNIST does not have a test data, so we need to manually process the
    training data to create test data.

    Returns
    -------
        DataLoader: The MNIST test set dataloader.
    """
    if "femnist" in dataset_config.path: #type: ignore
        dataset = load_dataset(path=dataset_config.path)["train"] #type: ignore
        split_dataset = dataset.train_test_split(  #type: ignore
            test_size=dataset_config.val_ratio, seed=dataset_config.seed #type: ignore
        )
        test_dataset = process_femnist(split_dataset["test"])
        test_dataset = test_dataset.with_transform(apply_transforms)
    else:
        test_dataset = load_dataset(path=dataset_config.path)["test"].with_transform( #type: ignore
            apply_transforms
        )
    return DataLoader(test_dataset, batch_size=dataset_config.batch_size) #type: ignore

import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader, Subset
import numpy as np

def create_heterogeneous_partition(dataset, num_clients=200, classes_per_client=2):
    """
    Create heterogeneous data partition following Li & Li 2023.
    Split into 400 shards (each containing samples from one class).
    Each client gets 2 shards from at most 2 classes.
    """
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(targets))

    class_indices = {c: np.where(targets == c)[0] for c in range(num_classes)}

    shards = []
    shards_per_class = 40

    for c in range(num_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)

        shard_size = len(indices) // shards_per_class
        for i in range(shards_per_class):
            start = i * shard_size
            end = start + shard_size if i < shards_per_class - 1 else len(indices)
            shards.append({
                'class': c,
                'indices': indices[start:end]
            })

    client_indices = []
    np.random.shuffle(shards)

    for client_id in range(num_clients):
        client_shards = shards[client_id * 2:(client_id + 1) * 2]

        indices = []
        for shard in client_shards:
            indices.extend(shard['indices'])

        np.random.shuffle(indices)
        client_indices.append(indices)

    return client_indices

def load_mnist_data(dataset_name="mnist", num_clients=200, batch_size=32):
    """Load MNIST or Fashion-MNIST with heterogeneous partitioning."""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) if dataset_name == "mnist"
                          else transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_name == "mnist":
        train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    else:  # fashion-mnist
        train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)

    client_indices = create_heterogeneous_partition(train_dataset, num_clients)

    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    return client_loaders, test_loader


# FEMNIST (natural per-writer partitioning via HuggingFace `flwrlabs/femnist`):
# 62-class FEMNIST with NaturalIdPartitioner on `writer_id`. We select a
# deterministic subset of writers (seeded) up to `num_clients`, then split each
# writer's samples 90/10 into (train, test). The held-out 10% is pooled into a
# single centralized test loader for server-side evaluation.
#
# Caches live in module globals so the dataset and the per-writer partitions
# are materialized exactly once per process (Flower re-invokes `client_fn`
# many times).

_FEMNIST_STATE = {
    "ready": False,
    "num_clients": None,
    "seed": None,
    "train_ds": None,          # HF Dataset, filtered to selected writers
    "test_ds": None,           # HF Dataset, pooled 10% holdout
    "partitions_train": None,  # List[List[int]] – indices into train_ds
    "writer_ids": None,        # List[str]
    "num_classes": 62,
}


FEMNIST_TRANSFORM = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,)),
])


def _pathological_partition(labels, num_clients, num_classes_per_client, rng):
    """Partition indices so each client gets exactly `num_classes_per_client` classes.

    Standard McMahan-style pathological partitioning: sort by label,
    divide into shards, assign shards round-robin to clients.
    """
    num_classes = int(labels.max()) + 1
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


def _dirichlet_partition(labels, num_clients, alpha, rng):
    """Partition indices using a Dirichlet(alpha) distribution over classes."""
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
        # Distribute remainder to clients with largest fractional parts.
        fracs = (proportions * len(idx_c)) - counts
        top_clients = np.argsort(fracs)[-remainder:]
        counts[top_clients] += 1
        offset = 0
        for k in range(num_clients):
            partitions[k].extend(idx_c[offset : offset + counts[k]].tolist())
            offset += counts[k]
    return partitions


def _ensure_femnist_loaded(config) -> None:
    """Load FEMNIST once, cache train/test splits and per-client partitions.

    Supports multiple partitioning modes via config.dataset.partitioning:
      - "natural"         : per-writer (NaturalIdPartitioner equivalent)
      - "pathological-1"  : 1 class per client
      - "pathological-2"  : 2 classes per client
      - "dirichlet"       : Dirichlet(alpha), alpha from config.dataset.alpha
    """
    num_clients = int(config["num_clients"])
    seed = int(config.get("dataset", {}).get("seed", config.get("seed", 42)))
    partitioning = str(
        config.get("dataset", {}).get("partitioning", "natural")
    )

    if (
        _FEMNIST_STATE["ready"]
        and _FEMNIST_STATE["num_clients"] == num_clients
        and _FEMNIST_STATE["seed"] == seed
        and _FEMNIST_STATE.get("partitioning") == partitioning
    ):
        return

    from datasets import load_dataset  # HF datasets

    print(
        f"[FEMNIST] Loading flwrlabs/femnist "
        f"(num_clients={num_clients}, seed={seed}, partitioning={partitioning}) ..."
    )
    full = load_dataset("flwrlabs/femnist", split="train")
    rng = np.random.default_rng(seed)

    if partitioning == "natural":
        writer_col = full["writer_id"]
        unique_writers = sorted(set(writer_col))

        if num_clients >= len(unique_writers):
            # Use every writer — skip the filter pass entirely.
            selected = list(unique_writers)
            filtered = full
            print(
                f"[FEMNIST] Using ALL {len(selected)} unique writers "
                f"(requested num_clients={num_clients})."
            )
        else:
            rng.shuffle(unique_writers)
            selected = unique_writers[:num_clients]
            selected_set = set(selected)
            print(
                f"[FEMNIST] {len(unique_writers)} unique writers in dataset; "
                f"keeping {len(selected)} for this run."
            )
            filtered = full.filter(
                lambda w: w in selected_set,
                input_columns=["writer_id"],
                num_proc=1,
            )

        writers_filtered = filtered["writer_id"]
        by_writer: dict = {}
        for i, w in enumerate(writers_filtered):
            by_writer.setdefault(w, []).append(i)

        partitions_train: list = []
        test_indices: list = []
        split_rng = np.random.default_rng(seed + 1)
        for w in selected:
            idx = np.array(by_writer.get(w, []), dtype=np.int64)
            if idx.size == 0:
                partitions_train.append([])
                continue
            split_rng.shuffle(idx)
            n_test = max(1, int(round(0.10 * len(idx))))
            test_indices.extend(idx[:n_test].tolist())
            partitions_train.append(idx[n_test:].tolist())

        train_ds = filtered
        test_ds = filtered.select(test_indices)

    else:
        # Artificial partitioning (pathological / dirichlet): global 90/10 split first.
        n_total = len(full)
        all_idx = np.arange(n_total)
        split_rng = np.random.default_rng(seed + 1)
        split_rng.shuffle(all_idx)
        n_test = max(1, int(round(0.10 * n_total)))
        test_idx = all_idx[:n_test].tolist()
        train_idx = all_idx[n_test:]

        train_ds = full.select(train_idx.tolist())
        test_ds = full.select(test_idx)

        train_labels = np.array(train_ds["character"])

        print(
            f"[FEMNIST] {n_total} total samples; "
            f"{len(train_idx)} train, {n_test} test."
        )

        if partitioning == "pathological-1":
            partitions_train = _pathological_partition(
                train_labels, num_clients, 1, rng
            )
        elif partitioning == "pathological-2":
            partitions_train = _pathological_partition(
                train_labels, num_clients, 2, rng
            )
        elif partitioning == "dirichlet":
            alpha = float(
                config.get("dataset", {}).get("alpha", 0.1)
            )
            partitions_train = _dirichlet_partition(
                train_labels, num_clients, alpha, rng
            )
            print(f"[FEMNIST] Dirichlet alpha={alpha}")
        else:
            raise ValueError(
                f"Unknown FEMNIST partitioning: {partitioning!r}. "
                f"Choose from: natural, pathological-1, pathological-2, dirichlet"
            )

        n_classes_per_client = []
        for part in partitions_train:
            if part:
                n_classes_per_client.append(
                    len(set(train_labels[part].tolist()))
                )
            else:
                n_classes_per_client.append(0)
        n_cls = np.array(n_classes_per_client)
        print(
            f"[FEMNIST] classes/client: "
            f"min={n_cls.min()}, median={int(np.median(n_cls))}, "
            f"mean={n_cls.mean():.1f}, max={n_cls.max()}"
        )

    test_classes = set(test_ds["character"])
    print(
        f"[FEMNIST] pooled test: {len(test_ds)} samples, "
        f"{len(test_classes)}/62 classes covered."
    )

    _FEMNIST_STATE.update(
        {
            "ready": True,
            "num_clients": num_clients,
            "seed": seed,
            "partitioning": partitioning,
            "train_ds": train_ds,
            "test_ds": test_ds,
            "partitions_train": partitions_train,
            "writer_ids": None,
        }
    )


def _femnist_collate(batch):
    """Collate an HF-style FEMNIST batch into (image_tensor, label_tensor)."""
    images = []
    labels = []
    for ex in batch:
        img = ex["image"]
        if not isinstance(img, torch.Tensor):
            img = FEMNIST_TRANSFORM(img)
        images.append(img)
        labels.append(int(ex["character"]))
    x = torch.stack(images, dim=0)  # (B, 1, 28, 28)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


class _HFIndexed(torch.utils.data.Dataset):
    """Thin wrapper around an HF Dataset + index list to give it __getitem__ semantics."""

    def __init__(self, hf_dataset, indices=None):
        self._ds = hf_dataset
        self._indices = indices  # None = all rows

    def __len__(self):
        return len(self._ds) if self._indices is None else len(self._indices)

    def __getitem__(self, i):
        j = i if self._indices is None else int(self._indices[i])
        return self._ds[j]


def _femnist_partition_loaders(partition_id: int, config) -> Tuple[DataLoader, DataLoader]:
    """Return (trainloader, valloader) for a single FEMNIST writer."""
    _ensure_femnist_loaded(config)
    batch_size = int(config["dataset"]["batch_size"])
    seed = int(config.get("dataset", {}).get("seed", config.get("seed", 42)))

    indices = _FEMNIST_STATE["partitions_train"][partition_id]  # type: ignore
    train_ds = _FEMNIST_STATE["train_ds"]                        # type: ignore

    # Split this writer's train indices into 80/20 train/val for local val.
    rng = np.random.default_rng(seed + 1000 + partition_id)
    idx = np.array(indices, dtype=np.int64)
    rng.shuffle(idx)
    n_val = max(1, int(round(0.20 * len(idx)))) if len(idx) > 1 else 0
    val_idx = idx[:n_val].tolist()
    trn_idx = idx[n_val:].tolist() if n_val > 0 else idx.tolist()

    trainset = _HFIndexed(train_ds, trn_idx)
    valset = _HFIndexed(train_ds, val_idx)

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_femnist_collate,
        drop_last=False,
    )
    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_femnist_collate,
        drop_last=False,
    )
    return trainloader, valloader


def _femnist_test_loader(run_config) -> DataLoader:
    """Return the centralized FEMNIST test loader (pooled across writers)."""
    _ensure_femnist_loaded(run_config)
    batch_size = int(run_config["dataset"].get("batch_size", 64))
    test_ds = _FEMNIST_STATE["test_ds"]
    return DataLoader(
        _HFIndexed(test_ds, None),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_femnist_collate,
        drop_last=False,
    )
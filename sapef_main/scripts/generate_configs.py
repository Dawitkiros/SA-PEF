"""Generate experiment configs for SA-PEF."""
from __future__ import annotations

from pathlib import Path


# scripts/<file>.py → repo (sapef_main/)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs"

CIFAR_SEEDS = [42, 43, 44, 45, 46]
FEMNIST_SEEDS = [42, 43, 44]
METHODS = {
    "fedavg": {"alpha": 0.0},
    "ef": {"alpha": 0.0},
    "saef": {"alpha": 1.0},
    "cser": {"alpha": 0.0},
    "sapef": {"alpha": 0.85},
}


def q(value: str) -> str:
    return '"' + value + '"'


def write_config(path: Path, values: dict, model: dict, dataset: dict, algorithm: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in values.items():
        if isinstance(value, str):
            rendered = q(value)
        elif isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, list):
            rendered = "[" + ", ".join(str(v) for v in value) + "]"
        else:
            rendered = str(value)
        lines.append(f"{key} = {rendered}")

    lines.extend(["", "[model]"])
    for key, value in model.items():
        lines.append(f"{key} = {q(value) if isinstance(value, str) else value}")

    lines.extend(["", "[dataset]"])
    for key, value in dataset.items():
        lines.append(f"{key} = {q(value) if isinstance(value, str) else value}")

    lines.extend(["", "[algorithm]", f"name = {q(algorithm)}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def common_cifar(
    *,
    dataset_name: str,
    model_name: str,
    num_classes: int,
    scenario: str,
    clients_per_round: int,
    dirichlet_alpha: float,
    sparsity: float,
    method: str,
    lr_max: float,
    lr_min: float,
) -> tuple[dict, dict, dict]:
    values = {
        "num-clients": 100,
        "num-server-rounds": 200,
        "local-work-mode": "steps",
        "local-steps": 5,
        "local-epochs": 5,
        "clients-per-round": clients_per_round,
        "fraction-fit": clients_per_round / 100,
        "min-available-clients": clients_per_round,
        "learning-rate-max": lr_max,
        "learning-rate-min": lr_min,
        "lr-warmup-rounds": 5,
        "learning-rate": lr_max,
        "global-lr": 1.0,
        "momentum": 0.9,
        "weight-decay": 5e-4,
        "comp-type": "topk",
        "sparsify-by": sparsity,
        "alpha-r": METHODS[method]["alpha"],
        "H": 5,
        "reset-frac": 0.1,
        "seed": CIFAR_SEEDS[0],
        "seed-list": ",".join(str(seed) for seed in CIFAR_SEEDS),
        "wandb-enabled": False,
        "wandb-project": "sapef",
        "wandb-name": f"{dataset_name}_{scenario}_{method}",
        "save-path": f"results/{dataset_name}/{scenario}/{method}/seed42/results.csv",
        "track-metrics": method in {"ef", "saef", "sapef"},
    }
    model = {"name": model_name, "num-classes": num_classes}
    dataset = {
        "name": dataset_name,
        "partitioning": "dirichlet",
        "alpha": dirichlet_alpha,
        "batch-size": 64,
        "seed": CIFAR_SEEDS[0],
        "val-ratio": 0.1,
    }
    return values, model, dataset


def generate_cifar() -> None:
    scenarios = {
        "cifar10": {
            "model": "ResNet9",
            "classes": 10,
            "lr_max": 0.1,
            "lr_min": 0.003,
            "items": {
                "p10_dir05_top001": (10, 0.5, 0.01),
                "p10_dir05_top010": (10, 0.5, 0.10),
                "p50_dir01_top001": (50, 0.1, 0.01),
                "p50_dir01_top010": (50, 0.1, 0.10),
            },
        },
        "cifar100": {
            "model": "ResNet18",
            "classes": 100,
            "lr_max": 0.1,
            "lr_min": 0.003,
            "items": {
                "p10_dir01_top001": (10, 0.1, 0.01),
                "p10_dir01_top010": (10, 0.1, 0.10),
                "p50_dir01_top001": (50, 0.1, 0.01),
                "p50_dir01_top010": (50, 0.1, 0.10),
            },
        },
    }
    for dataset_name, spec in scenarios.items():
        for scenario, (clients_per_round, alpha, sparsity) in spec["items"].items():
            for method in METHODS:
                values, model, dataset = common_cifar(
                    dataset_name=dataset_name,
                    model_name=spec["model"],
                    num_classes=spec["classes"],
                    scenario=scenario,
                    clients_per_round=clients_per_round,
                    dirichlet_alpha=alpha,
                    sparsity=sparsity,
                    method=method,
                    lr_max=spec["lr_max"],
                    lr_min=spec["lr_min"],
                )
                write_config(
                    OUT / dataset_name / scenario / f"{method}.toml",
                    values,
                    model,
                    dataset,
                    method,
                )


def generate_alpha_sweep() -> None:
    for dataset_name, model_name, num_classes, lr_max in [
        ("cifar10", "ResNet9", 10, 0.1),
        ("cifar100", "ResNet18", 100, 0.1),
    ]:
        for idx in range(11):
            alpha = idx / 10.0
            alpha_name = str(alpha).replace(".", "_")
            values, model, dataset = common_cifar(
                dataset_name=dataset_name,
                model_name=model_name,
                num_classes=num_classes,
                scenario=f"alpha_{alpha_name}",
                clients_per_round=10,
                dirichlet_alpha=0.1,
                sparsity=0.01,
                method="sapef",
                lr_max=lr_max,
                lr_min=0.003,
            )
            values["alpha-r"] = alpha
            values["wandb-name"] = f"{dataset_name}_alpha_{alpha_name}"
            values["save-path"] = (
                f"results/alpha_sweep/{dataset_name}/alpha_{alpha_name}/"
                "seed42/results.csv"
            )
            write_config(
                OUT / "alpha_sweep" / dataset_name / f"alpha_{alpha_name}.toml",
                values,
                model,
                dataset,
                "sapef",
            )


def generate_gradient_mismatch() -> None:
    for dataset_name, model_name, num_classes, lr_max in [
        ("cifar10", "ResNet9", 10, 0.1),
        ("cifar100", "ResNet18", 100, 0.1),
    ]:
        for method in ["ef", "saef", "sapef", "fedavg", "cser"]:
            values, model, dataset = common_cifar(
                dataset_name=dataset_name,
                model_name=model_name,
                num_classes=num_classes,
                scenario="gradient_mismatch",
                clients_per_round=10,
                dirichlet_alpha=0.1,
                sparsity=0.01,
                method=method,
                lr_max=lr_max,
                lr_min=0.003,
            )
            values["track-metrics"] = True
            values["save-path"] = (
                f"results/gradient_mismatch/{dataset_name}/{method}/"
                "seed42/results.csv"
            )
            write_config(
                OUT / "gradient_mismatch" / dataset_name / f"{method}.toml",
                values,
                model,
                dataset,
                method,
            )


def generate_femnist() -> None:
    for scenario, partitioning in [
        ("natural_top001", "natural"),
        ("patho2_top001", "pathological-2"),
    ]:
        for method in METHODS:
            values = {
                "num-clients": 200,
                "num-server-rounds": 200,
                "local-work-mode": "epochs",
                "local-epochs": 5,
                "local-steps": 10,
                "clients-per-round": 20,
                "fraction-fit": 0.1,
                "min-available-clients": 20,
                "learning-rate-max": 0.01,
                "learning-rate-min": 0.001,
                "lr-warmup-rounds": 5,
                "learning-rate": 0.01,
                "global-lr": 1.0,
                "momentum": 0.9,
                "weight-decay": 1e-4,
                "comp-type": "topk",
                "sparsify-by": 0.01,
                "alpha-r": METHODS[method]["alpha"],
                "H": 5,
                "reset-frac": 0.1,
                "seed": FEMNIST_SEEDS[0],
                "seed-list": ",".join(str(seed) for seed in FEMNIST_SEEDS),
                "wandb-enabled": False,
                "wandb-project": "sapef_femnist",
                "wandb-name": f"femnist_{scenario}_{method}",
                "save-path": f"results/femnist/{scenario}/{method}/seed42/results.csv",
                "track-metrics": method in {"ef", "saef", "sapef"},
            }
            model = {"name": "FEMNIST_CNN", "num-classes": 62}
            dataset = {
                "name": "femnist",
                "partitioning": partitioning,
                "batch-size": 32,
                "val-ratio": 0.1,
                "seed": FEMNIST_SEEDS[0],
            }
            write_config(
                OUT / "femnist" / scenario / f"{method}.toml",
                values,
                model,
                dataset,
                method,
            )


def generate_quick_test() -> None:
    for method in METHODS:
        values, model, dataset = common_cifar(
            dataset_name="cifar10",
            model_name="ResNet9",
            num_classes=10,
            scenario="quick-test",
            clients_per_round=2,
            dirichlet_alpha=0.5,
            sparsity=0.10,
            method=method,
            lr_max=0.01,
            lr_min=0.001,
        )
        values.update(
            {
                "num-clients": 4,
                "num-server-rounds": 2,
                "local-steps": 1,
                "local-epochs": 1,
                "fraction-fit": 0.5,
                "min-available-clients": 2,
                "seed-list": "42",
                "wandb-name": f"quick_test_cifar10_{method}",
                "save-path": f"results/quick-test/{method}_cifar10_tiny/results.csv",
                "track-metrics": False,
            }
        )
        dataset["batch-size"] = 16
        write_config(
            OUT / "quick-test" / f"{method}_cifar10_tiny.toml",
            values,
            model,
            dataset,
            method,
        )


SCAFCOM_LRS = {
    # (global_lr, local_lr) per method for the CIFAR-10/SCAFCOM ablation table.
    "fedavg":  (1.0, 1e-1),
    "ef":      (1.0, 1e-3),
    "saef":    (1.0, 1e-3),
    "sapef":   (1.0, 1e-3),
    "scafcom": (3.0, 1e-1),
    "scaffold": (1.0, 1e-1),
}


def _ablation_methods_alpha(method: str) -> float:
    if method == "fedavg":
        return 0.0
    if method == "saef":
        return 1.0
    if method == "sapef":
        return 0.85
    return 0.0


def generate_mnist_scafcom() -> None:
    """MNIST/2-layer FC, N=200, pathological-2, p=0.1, T=10, Top-1%."""
    methods = ["fedavg", "ef", "sapef", "scafcom", "scaffold"]
    for method in methods:
        global_lr, local_lr = SCAFCOM_LRS[method]
        sparsity = 0.0 if method in {"fedavg", "scaffold"} else 0.01
        comp_type = "none" if method in {"fedavg", "scaffold"} else "topk"
        values = {
            "num-clients": 200,
            "num-server-rounds": 200,
            "local-work-mode": "steps",
            "local-steps": 10,
            "local-epochs": 5,
            "clients-per-round": 20,
            "fraction-fit": 0.1,
            "min-available-clients": 20,
            "learning-rate-max": local_lr,
            "learning-rate-min": local_lr,
            "lr-warmup-rounds": 0,
            "learning-rate": local_lr,
            "global-lr": global_lr,
            "momentum": 0.0,
            "weight-decay": 0.0,
            "comp-type": comp_type,
            "sparsify-by": sparsity,
            "alpha-r": _ablation_methods_alpha(method),
            "alpha-sc": 0.1,
            "beta-sc": 0.2,
            "H": 5,
            "reset-frac": 0.1,
            "seed": FEMNIST_SEEDS[0],
            "seed-list": ",".join(str(s) for s in FEMNIST_SEEDS),
            "wandb-enabled": False,
            "wandb-project": "sapef_ablations",
            "wandb-name": f"mnist_scafcom_{method}",
            "save-path": f"results/ablations/mnist_scafcom/{method}/seed42/results.csv",
            "track-metrics": False,
        }
        model = {"name": "TwoLayerMLP", "num-classes": 10}
        dataset = {
            "name": "mnist",
            "partitioning": "shards",
            "batch-size": 32,
            "val-ratio": 0.1,
            "seed": FEMNIST_SEEDS[0],
        }
        write_config(
            OUT / "ablations" / "mnist_scafcom" / f"{method}.toml",
            values,
            model,
            dataset,
            method,
        )


def generate_cifar10_scafcom(sparsity: float, scenario: str) -> None:
    """CIFAR-10/ResNet-9, K=100, R=200, p=0.1 (10/round), T=20 local steps."""
    methods = ["fedavg", "ef", "saef", "sapef", "scafcom"]
    for method in methods:
        global_lr, local_lr = SCAFCOM_LRS[method]
        comp_type = "none" if method == "fedavg" else "topk"
        sparsity_eff = 0.0 if method == "fedavg" else sparsity
        values = {
            "num-clients": 100,
            "num-server-rounds": 200,
            "local-work-mode": "steps",
            "local-steps": 20,
            "local-epochs": 5,
            "clients-per-round": 10,
            "fraction-fit": 0.1,
            "min-available-clients": 10,
            "learning-rate-max": local_lr,
            "learning-rate-min": local_lr,
            "lr-warmup-rounds": 0,
            "learning-rate": local_lr,
            "global-lr": global_lr,
            "momentum": 0.0,
            "weight-decay": 5e-4,
            "comp-type": comp_type,
            "sparsify-by": sparsity_eff,
            "alpha-r": _ablation_methods_alpha(method),
            "alpha-sc": 0.1,
            "beta-sc": 0.2,
            "H": 5,
            "reset-frac": 0.1,
            "seed": CIFAR_SEEDS[0],
            "seed-list": ",".join(str(s) for s in CIFAR_SEEDS),
            "wandb-enabled": False,
            "wandb-project": "sapef_ablations",
            "wandb-name": f"cifar10_scafcom_{scenario}_{method}",
            "save-path": f"results/ablations/cifar10_scafcom_{scenario}/{method}/seed42/results.csv",
            "track-metrics": False,
        }
        model = {"name": "ResNet9", "num-classes": 10}
        dataset = {
            "name": "cifar10",
            "partitioning": "dirichlet",
            "alpha": 0.5,
            "batch-size": 64,
            "seed": CIFAR_SEEDS[0],
            "val-ratio": 0.1,
        }
        write_config(
            OUT / "ablations" / f"cifar10_scafcom_{scenario}" / f"{method}.toml",
            values,
            model,
            dataset,
            method,
        )


def generate_ablations() -> None:
    generate_mnist_scafcom()
    generate_cifar10_scafcom(sparsity=0.01, scenario="top001")
    generate_cifar10_scafcom(sparsity=0.10, scenario="top010")


def main() -> None:
    generate_cifar()
    generate_alpha_sweep()
    generate_gradient_mismatch()
    generate_femnist()
    generate_quick_test()
    generate_ablations()


if __name__ == "__main__":
    main()

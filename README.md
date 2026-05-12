# SA-PEF: Step-Ahead Partial Error Feedback

Official implementation for our paper **Step-Ahead Partial Error Feedback for Communication-Efficient Federated Learning**.

## Installation

Conda / mamba:

```bash
cd SA-PEF
mamba env create -f sapef_env.yml      # or: conda env create -f sapef_env.yml
conda activate sapef
pip install -e sapef_main               # install the local package itself
```

Or with plain pip (Python 3.11+):

```bash
cd SA-PEF/sapef_main
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[tracking]"            # drop [tracking] if you don't want wandb
```

GPU runs use the default PyTorch wheel from PyPI; if your CUDA toolkit needs a
different build, install torch from the appropriate index URL before
`pip install -e .`

## Datasets

CIFAR-10 and CIFAR-100 are loaded through `torchvision` and are cached under `sapef_main/data/` by default. FEMNIST is loaded through `datasets` from `flwrlabs/femnist`; for clusters, set `HF_HOME` to a writable cache directory.

```bash
cd SA-PEF/sapef_main
python scripts/prefetch_femnist.py
```

Generated data, checkpoints, logs, W&B runs, and result files are ignored by `.gitignore`.

## Federations

`flwr run` requires a named federation. Per the migration notice in `pyproject.toml`,
federation definitions live in your local Flower configuration file rather than in
the project's `pyproject.toml`. Add the following to `~/.config/flwr/federations.toml`
(or the equivalent for your platform) before running the commands below:

```toml
[federations.quick-test]
options.num-supernodes = 4
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 0.0

[federations.gpu-simulation]
options.num-supernodes = 100
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 0.05
```

Adjust `num-gpus` per client to match your hardware.

## Quick Test

Use the small 4-client `quick-test` federation:

```bash
cd SA-PEF/sapef_main
for cfg in configs/quick-test/*.toml; do
  flwr run . quick-test -c "$cfg"
done
```

This checks that FedAvg, EF, SAEF, SA-PEF, and CSER launch, write result CSVs, and exercise sparse residual paths.

## Main Runs

```bash
cd SA-PEF/sapef_main
for cfg in configs/cifar10/*/*.toml configs/cifar100/*/*.toml; do
  flwr run . gpu-simulation -c "$cfg"
done
```

Additional suites:

```bash
for cfg in configs/alpha_sweep/{cifar10,cifar100}/*.toml; do
  flwr run . gpu-simulation -c "$cfg"
done

for cfg in configs/femnist/*/*.toml; do
  flwr run . gpu-simulation -c "$cfg"
done

for cfg in configs/ablations/*/*.toml; do
  flwr run . gpu-simulation -c "$cfg"
done
```

Each config writes to `sapef_main/results/.../results.csv`. The configs include `seed-list` metadata; run additional seeds by copying or overriding `seed`, `dataset.seed`, `save-path`, and `wandb-name`.

## Figures

```bash
cd SA-PEF/sapef_main
python scripts/plot_curves.py --manifest configs/figures.yaml --all
python scripts/plot_alpha_sweep.py --manifest configs/figures.yaml
```

Figures are written to `sapef_main/figures/`. Communication plots show cumulative uplink payload (per-round metric is bytes/1024^2 = MiB, summed and divided by 1024 to GiB; the axis is labelled "GB").

## Figure Mapping

| Result              | Configs                                                 | Plot command                                             |
| ------------------- | ------------------------------------------------------- | -------------------------------------------------------- |
| CIFAR-10 main       | `configs/cifar10/*/*.toml`                              | `python scripts/plot_curves.py --figure cifar10_main`    |
| CIFAR-100 main      | `configs/cifar100/*/*.toml`                             | `python scripts/plot_curves.py --figure cifar100_main`   |
| α ablation          | `configs/alpha_sweep/{cifar10,cifar100}/*.toml`         | `python scripts/plot_alpha_sweep.py`                     |
| FEMNIST             | `configs/femnist/*/*.toml`                              | `python scripts/plot_curves.py --figure femnist`         |
| MNIST vs SCAFCOM    | `configs/ablations/mnist_scafcom/*.toml`                | `python scripts/plot_curves.py --figure mnist_scafcom`   |
| CIFAR-10 vs SCAFCOM | `configs/ablations/cifar10_scafcom_top{001,010}/*.toml` | `python scripts/plot_curves.py --figure cifar10_scafcom` |

See `sapef_main/configs/figures.yaml` for the full mapping.

## Implemented Methods

Main configs cover FedAvg, EF, SAEF, CSER, and SA-PEF. Ablation configs additionally cover SCAFFOLD and SCAFCOM (control-variate baselines). CIFAR main configs use fixed local SGD steps via `local-work-mode = "steps"` and `local-steps = 5`; FEMNIST uses `local-work-mode = "epochs"` and `local-epochs = 5`. The CIFAR-10/SCAFCOM ablation uses `local-steps = 20`; the MNIST/SCAFCOM ablation uses `local-steps = 10`.

## Citation

```bibtex
@article{redie2026sa,
  title={{SA-PEF}: Step-Ahead Partial Error Feedback for Efficient Federated Learning},
  author={Redie, Dawit Kiros and Arablouei, Reza and Werner, Stefan},
  journal={arXiv preprint arXiv:2601.20738},
  year={2026}
}
```

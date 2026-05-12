"""Unified Server App for multiple FL algorithms."""
import json
import os
import torch
from typing import Callable, Dict, Optional, Tuple, List

from flwr.server import ServerApp, ServerAppComponents, ServerConfig, SimpleClientManager
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

from .strategies.fedspars import FedSpars
from .strategies.cser import CSERStrategy
from .strategies.strategy_fedvg_straggler import FedAvgWithStragglerDrop
from .strategies.scaffold_like import ScaffoldLike
from .dataset import prepare_test_loader, load_test_data, load_mnist_data
from .models import instantiate_model, set_weights, set_weights_with_dtype_handling, get_weights, test, test_mnist, test_mnist_scaffold
from .server import ResultsSaverServer, history_saver
from .utils import context_to_easydict


def gen_evaluate_fn_fedprox(
    testloader: DataLoader,
    device: torch.device,
    run_config: dict,
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """Generate evaluation function for FedProx/FedAvg."""
    
    def evaluate(
        server_round: int,
        parameters_ndarrays: NDArrays,
        config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        net = instantiate_model(run_config)
        set_weights(net, parameters_ndarrays)
        net.to(device)
        if run_config["dataset"]["name"].lower() == "mnist_old":
            loss, accuracy = test_mnist(net, testloader, device=device)
        else:
            loss, accuracy = test(net, testloader, device=device)
        if run_config.get("wandb_enabled", False):
            try:
                import wandb
                wandb.log({"acc": accuracy, "loss": loss}, step=server_round)
            except ImportError:
                pass
        return loss, {"accuracy": accuracy}
    
    return evaluate


def gen_evaluate_fn_fedspars(
    testloader: DataLoader,
    device: torch.device,
    run_config: dict,
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """Generate evaluation function for FedSpars with WandB support."""
    
    def evaluate(
        server_round: int,
        parameters_ndarrays: NDArrays,
        config_dict: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        net = instantiate_model(run_config)
        net.to(device)
        set_weights_with_dtype_handling(net, parameters_ndarrays)
        
        ds_name = run_config["dataset"]["name"].lower()
        if ds_name in ["mnist", "fashion-mnist"]:
            loss, accuracy = test_mnist_scaffold(net, testloader, device=device)
        else:
            loss, accuracy = test(net, testloader, device=device)

        if run_config.get("wandb_enabled", False):
            try:
                import wandb
                wandb.log({"acc": accuracy, "loss": loss}, step=server_round)
            except ImportError:
                pass
        
        return loss, {"accuracy": accuracy}
    
    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average of accuracy metric."""
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def create_fedprox_strategy(
    run_config: dict,
    parameters,
    evaluate_fn: Callable
):
    """Create FedAvg/FedProx strategy."""
    
    def get_on_fit_config():
        def fit_config_fn(server_round: int):
            return {"current_round": server_round}

        return fit_config_fn
    
    return FedAvgWithStragglerDrop(
        fraction_fit=float(run_config.fraction_fit),
        fraction_evaluate=run_config.fraction_evaluate,
        min_available_clients=run_config.min_available_clients,
        initial_parameters=parameters,
        on_fit_config_fn=get_on_fit_config(),
        evaluate_fn=evaluate_fn,
    )


def create_fedspars_strategy(
    run_config: dict,
    parameters,
    evaluate_fn: Callable,
    net: torch.nn.Module,
):
    """Create FedSpars strategy."""
    return FedSpars(
        fraction_fit=0.0000001,  # Use min_fit_clients instead
        fraction_evaluate=0.0,
        min_fit_clients=run_config.clients_per_round,
        min_available_clients=run_config.clients_per_round,
        min_evaluate_clients=0,
        initial_parameters=parameters,
        evaluate_fn=evaluate_fn,
        net=net,
        config=run_config,
    )


def create_cser_strategy(
    run_config: dict,
    parameters,
    evaluate_fn: Callable,
    net: torch.nn.Module,
):
    """Create CSERStrategy (FedSpars + periodic error-reset digest)."""
    return CSERStrategy(
        fraction_fit=0.0000001,  # Use min_fit_clients instead
        fraction_evaluate=0.0,
        min_fit_clients=run_config.clients_per_round,
        min_available_clients=run_config.clients_per_round,
        min_evaluate_clients=0,
        initial_parameters=parameters,
        evaluate_fn=evaluate_fn,
        net=net,
        config=run_config,
        H=int(run_config.get("H", 5)),
        reset_frac=float(run_config.get("reset_frac", 0.1)),
    )

def server_fn(context: Context):
    """Unified server function that creates appropriate strategy based on config."""

    configs = context_to_easydict(context)
    run_config = configs.run_config

    algorithm = run_config.algorithm.name.lower()
    num_rounds = int(run_config.num_server_rounds)

    print("=" * 60)
    print("Federated Learning Experiment Config")
    print("=" * 60)
    print(json.dumps(run_config, indent=4))
    print("=" * 60)
    print(f"Algorithm: {algorithm}")
    print(f"Rounds: {num_rounds}")

    net = instantiate_model(run_config)
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)

    if run_config["dataset"]["name"].lower() == "mnist_prox":
        testloader = prepare_test_loader(run_config.dataset)
        evaluate_fn = gen_evaluate_fn_fedprox(testloader, device, run_config)
    else:
        testloader = load_test_data(run_config)
        evaluate_fn = gen_evaluate_fn_fedspars(testloader, device, run_config)
    if run_config.get("wandb_enabled", False):
        try:
            import wandb
            wandb.init(
                project=run_config.get("wandb_project", "federated-learning"),
                name=run_config.get("wandb_name", f"{algorithm}_experiment"),
                config=run_config
            )
            print("[Info] WandB initialized successfully")
        except ImportError:
            print("[Warning] WandB not installed, skipping WandB logging")
    
    if algorithm in ["sapef", "saef", "ef"]:
        save_dir = f"checkpoints/{algorithm}/{run_config.comp_type}_{run_config.sparsify_by}_{run_config.dataset.name}_{run_config.dataset.partitioning}_{run_config.num_clients}_{run_config.clients_per_round}_{run_config.alpha_r}_{run_config.learning_rate}_{run_config.seed}"
        os.makedirs(save_dir, exist_ok=True)
        net.to(device)
        torch.save(net.state_dict(), f"{save_dir}/round_0.pth")
        print(f"[Info] Saved initial checkpoint to {save_dir}/round_0.pth")

        strategy = create_fedspars_strategy(
            run_config=run_config,
            parameters=parameters,
            evaluate_fn=evaluate_fn,
            net=net
        )
    elif algorithm == "cser":
        # CSER uses a distinct checkpoint dir that also encodes H / reset_frac
        # so different CSER hyperparameter settings don't collide.
        H = int(run_config.get("H", 5))
        reset_frac = float(run_config.get("reset_frac", 0.1))
        save_dir = (
            f"checkpoints/cser/{run_config.comp_type}_{run_config.sparsify_by}_"
            f"{run_config.dataset.name}_{run_config.dataset.partitioning}_"
            f"{run_config.num_clients}_{run_config.clients_per_round}_"
            f"{run_config.alpha_r}_{run_config.learning_rate}_"
            f"{run_config.seed}_H{H}_rf{reset_frac}"
        )
        os.makedirs(save_dir, exist_ok=True)
        net.to(device)
        torch.save(net.state_dict(), f"{save_dir}/round_0.pth")
        print(f"[Info] Saved initial CSER checkpoint to {save_dir}/round_0.pth")

        strategy = create_cser_strategy(
            run_config=run_config,
            parameters=parameters,
            evaluate_fn=evaluate_fn,
            net=net,
        )
    elif algorithm in ["fedavg", "fedprox"]:
        strategy = create_fedprox_strategy(
            run_config=run_config,
            parameters=parameters,
            evaluate_fn=evaluate_fn
        )
    elif algorithm in ["scaffold", "scallion", "scafcom"]:
        strategy = ScaffoldLike(
            net=net,
            test_loader=testloader,
            run_config=run_config,
            evaluate_fn=evaluate_fn,
            fraction_fit=float(run_config.fraction_fit),
            fraction_evaluate=float(run_config.fraction_evaluate),
            min_fit_clients=int(run_config.clients_per_round),
            min_evaluate_clients=int(run_config.get("min_evaluate_clients", 0)),
            min_available_clients=int(run_config.min_available_clients),
        )
    else:
        raise ValueError(
            f"Unknown algorithm: '{algorithm}'. "
            f"Available: ['fedavg', 'fedprox', 'sapef', 'saef', 'ef', 'cser', "
            f"'scaffold', 'scallion', 'scafcom']"
        )
    
    client_manager = SimpleClientManager()
    server = ResultsSaverServer(
        client_manager=client_manager,
        strategy=strategy,
        results_saver_fn=history_saver,
        run_config=run_config,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(server=server, config=config)


app = ServerApp(server_fn=server_fn)
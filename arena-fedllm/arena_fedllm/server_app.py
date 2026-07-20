"""arena-llm: ARENA Flower / FlowerTune app for simulating LLM fine-tuning."""

import os
from datetime import datetime

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAvgM, FedMedian, FedProx
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from arena_fedllm.dataset import replace_keys
from arena_fedllm.models import get_model
from arena_fedllm.fedavgopt import FedAvgOpt

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Get initial model weights
    init_model = get_model(cfg.model)
    arrays = ArrayRecord(get_peft_model_state_dict(init_model))

    # Define strategy
    strategy_name = str(cfg.strategy.name)
    strategy_kwargs = {
        "fraction_train": cfg.strategy.fraction_train,
        "fraction_evaluate": cfg.strategy.fraction_evaluate,
    }

    if strategy_name == "Federated Averaging (FedAvg)" or strategy_name == "FedAvg":
        strategy = FedAvg(**strategy_kwargs)
    elif strategy_name == "Federated Median (FedMedian)":
        strategy = FedMedian(**strategy_kwargs)
    elif strategy_name == "Federated Averaging with Momentum (FedAvgM)":
        strategy = FedAvgM(
            **strategy_kwargs,
            server_learning_rate=cfg.strategy.fedavgm.server_learning_rate,
            server_momentum=cfg.strategy.fedavgm.server_momentum,
        )
    elif strategy_name == "FedProx":
        strategy = FedProx(
            **strategy_kwargs,
            proximal_mu=cfg.strategy.fedprox.mu,
        )
    elif strategy_name == "FedAvgOpt":
        strategy = FedAvgOpt(**strategy_kwargs)
    else:
        strategy = FedAvg(**strategy_kwargs)

    # Start strategy:
    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"save_path": save_path}),
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(
            cfg.model, cfg.train.save_every_round, num_rounds, save_path
        ),
    )


# Function user for saving global model checkpoints
def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path):
    """Return an evaluation function for saving global model."""

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        # Save model
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Init model
            model = get_model(model_cfg)
            set_peft_model_state_dict(model, arrays.to_torch_state_dict())

            model.save_pretrained(f"{save_path}/peft_{server_round}")

        return MetricRecord()

    return evaluate

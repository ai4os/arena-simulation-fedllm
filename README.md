# ARENA FlowerTune app for simulating the fine-tuning of LLMs

Simulation-mode version of the federated fine-tuning app based on FloweTune. This FlowerTune LLM app (`client_app.py`, `server_app.py`, `dataset.py`, `models.py`) runs with Flower's **simulation** federation: the server and every client run as processes on the **same machine**, inside the same `flwr run` invocation.

The base model and the dataset are both fully configurable by the user: this app is not tied to any specific model or dataset. This is selected in the `MODEL_NAME` and `DATA_FILE_NAME` environment variables which are set when launching the service from the [dashboard](https://dashboard.cloud.ai4eosc.eu).

> **WARNING: SIMULATION ONLY**
>
> This setup is for local testing, development, and experimentation. It is **not** a federated learning deployment and does not provide the
> guarantees a real one would:
>
> - The server and all "clients" run on the same machine, there is no real separation between them.
> - All clients read from the **same central data** and  get a random, non-overlapping partition. There is no data
>   isolation between clients.
> - There is no TLS, no authentication, and no transport-layer security (not applicable here).
> - Do not use this mode to validate any privacy, security, or
>   data-isolation property of your system. 
> - Launch the server and the clients in different machines with TLS-secured connections if you want to use private data in each client. 

## Why use the simulation model?
 
- **Quick tests before requesting real resources**: verify that your model, dataset, or LoRA configuration actually converges during training and produces the expected results before proceeding to provision GPUs on multiple physical machines (or in a cluster), which is slower and more complex to set up.
- **Measure distribution efficiency**: get a first read on aggregation   overhead, per-round timing, and how training loss evolves as you change  `num-server-rounds` or the number of simulated clients.
- **Test data-partitioning strategies**: try different splits (more/fewer partitions, IID vs. eterogeneous client datasets) and see how different aggregation stragies perform, without needing several physical sites connected.
- **Fast iteration**: check that `client_app.py`/`server_app.py` logic, LoRA config, and training arguments actually work end to end, without oordinating multiple nodes.
- **Hyperparameter exploration**: try different `num-server-rounds`,  learning rates, LoRA rank/alpha, batch sizes, sequence lengths, etc.,  and see how ifferent federated aggregation stragies behave before moving to a real distirbuted deployment with TLS configuration.
- **Debugging**: reproduce a training issue quickly without coordinating multiple physical machines.
- **Resource sizing**: Get a general idea of the VRAM and processing time required per client for the model size and quantization settings you've chosen, so you can plan the resources needed. 

## Launch the simulation
The enviroment variables, including the path to the data (data have to be uploaded in the deployment) the model to be fine-tuned, the hyperparameters and the HuggingFace token, are introduced via the dashboard when configuring the service. Then, open a terminal and run: 

```bash
cd arena-fedllm
flwr run . -- stream
```
 
## How data is distributed
 
By default, a single CSV file (set via `DATA_FILE_NAME`, i.e.
`dataset.name`) is split into `num-supernodes` IID partitions using `flwr_datasets`' `IidPartitioner`. Each simulated client is assigned one partition by `partition-id` and only trains on that slice.

## Environment variables to be configured
 
| Environment variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | `---` | HuggingFace token |
| `PUBLISHER_NAME` | `user_arena` | Publisher name (user, optional) |
| `DATA_FILE_NAME` | `arena_fedllm/eosc_dataset.csv` | Path to the training data |
| `MODEL_NAME` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Base model |
| `MODEL_QUANTIZATION` | `4` | Quantization bits (4 or 8) |
| `PEFT_LORA_R` | `32` | LoRA rank |
| `PEFT_LORA_ALPHA` | `64` | LoRA alpha |
| `SAVE_EVERY_ROUND` | `5` | Save a checkpoint every N rounds |
| `LEARNING_RATE_MAX` | `5e-5` | Cosine schedule peak LR |
| `LEARNING_RATE_MIN` | `1e-6` | Cosine schedule floor LR |
| `SEQ_LENGTH` | `512` | Max training sequence length |
| `NUM_TRAIN_EPOCHS` | `3` | Local epochs per client per round |
| `SAVE_TOTAL_LIMIT` | `10` | Max checkpoints kept |
| `FRACTION_TRAIN` | `0.1` | Fraction of clients sampled per round for training |
| `FRACTION_EVALUATE` | `0.0` | Fraction of clients sampled per round for evaluation |
| `FEDERATED_STRATEGY` | `Federated Averaging (FedAvg)` | Aggregation strategy used |
| `FEDPROX_MU` | `0.01` | Proximal term weight (only used if `FEDERATED_STRATEGY` is FedProx) |
| `FEDAVGM_SERVER_LEARNING_RATE` | `0.1` | Server-side LR (only used if `FEDERATED_STRATEGY` is FedAvgM) |
| `FEDAVGM_SERVER_MOMENTUM` | `0.9` | Server-side momentum (only used if `FEDERATED_STRATEGY` is FedAvgM) |
| `NUM_RUNS` | `100` | Number of federated rounds |


### Warning
This project is under active development. 

## License
This project is licensed under the [Apache 2.0 license](https://github.com/ai4os/arena-simulation-fedllm/blob/main/LICENSE).

## Funding and acknowledgments
This work is funded by European Union through the EOSC-ARENA project (Horizon Europe) under Grant number [101292597](https://cordis.europa.eu/project/id/101292597).
<p>
<img align="center" width="250" src="https://raw.githubusercontent.com/AI4EOSC/.github/ai4eosc/profile/EN-Funded.jpg">
<img align="center" width="350" src="https://ai4eosc.eu/_astro/EOSC-ARENA.DElvfRq2.png">
<p>

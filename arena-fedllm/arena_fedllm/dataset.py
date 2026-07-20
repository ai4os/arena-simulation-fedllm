"""arena-fedllm: A Flower / FlowerTune app."""

import os

from datasets import load_dataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

FDS = None  # Cache FederatedDataset


def formatting_prompts_func(example):
    """Construct prompts from instruction/input/response columns."""
    output_texts = []
    mssg = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )
    instructions = example.get("instruction", [])
    responses = example.get("response", [])
    inputs = example.get("input", [""] * len(instructions))

    for instruction, response, user_input in zip(instructions, responses, inputs):
        instruction_text = instruction.strip()
        if user_input:
            instruction_text = f"{instruction_text}\n{user_input}".strip()
        text = (
            f"{mssg}\n### Instruction:\n{instruction_text}\n"
            f"### Response: {response}"
        )
        output_texts.append(text)
    return output_texts


def get_tokenizer_and_data_collator_and_propt_formatting(model_name: str):
    """Get tokenizer, data_collator and prompt formatting."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    response_template_with_context = "\n### Response:"  # alpaca response tag
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    return tokenizer, data_collator, formatting_prompts_func


def load_data(partition_id: int, num_partitions: int, dataset_path: str):
    """Load partition data from a local CSV file."""
    global FDS
    if FDS is None:
        resolved_path = dataset_path
        if not os.path.exists(resolved_path):
            resolved_path = os.path.join(os.getcwd(), dataset_path)
        dataset = load_dataset("csv", data_files=resolved_path, split="train")
        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = dataset
        FDS = partitioner
    return FDS.load_partition(partition_id=partition_id)


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict

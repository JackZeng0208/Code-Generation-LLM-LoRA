# Reference: https://github.com/Lightning-AI/lit-llama/blob/main/scripts/prepare_alpaca.py
# Reference: https://github.com/tloen/alpaca-lora
import torch
import requests
import json
from torch.utils.data import random_split
from transformers import LlamaTokenizer
from tqdm import tqdm
from pathlib import Path

IGNORE_INDEX = -1

def prepare(
    file_path: Path = Path("Code-Generation-LLM-LoRA/data/lora_fine_tuning_data.json"),
    test_split_size: int = 2000,
    max_seq_length: int = 512,
    seed: int = 42,
    mask_inputs: bool = False,
) -> None:
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    data = []
    with open(file_path) as file:
        data = json.load(file)

    print(f"Loaded {len(data)} samples")
    train_split_size = len(data) - test_split_size
    train_set, test_set = random_split(
        data, 
        lengths=(train_split_size, test_split_size),
        generator=torch.Generator().manual_seed(seed),
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set)} samples")
    print(f"val has {len(test_set)} samples")

    print("Processing train split ...")
    train_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_set)]
    torch.save(train_set, file_path.parent / "train.pt")

    print("Processing test split ...")
    test_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_set)]
    torch.save(test_set, file_path.parent / "test.pt")



def prepare_sample(example: dict, tokenizer: LlamaTokenizer, max_length: int, mask_inputs: bool = True):
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["answer"]
    encoded_full_prompt = tokenizer.encode(full_prompt, add_special_tokens=False, max_length=max_length, truncation=True, truncation_strategy="only_second")
    encoded_full_prompt_and_response  = tokenizer.encode(full_prompt_and_response, add_special_tokens=True, max_length=max_length)
    # print(type(encoded_full_prompt_and_response))
    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.copy()
    if mask_inputs:
        labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    return {**example, "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt, "labels": labels}

def generate_prompt(example):
    return (
        "Below is an instruction that describes a task. "
        "You are an AI program assistant. Your task is to solve programming problems from interviews and coding contests only using Python.\n\n"
        f"### Instruction:\n{example['question']}\n\n### Response:"
    )

if __name__ == "__main__":
    prepare(mask_inputs=True)
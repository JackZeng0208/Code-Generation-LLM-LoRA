from datasets import load_dataset
import json
import random

dataset = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1")
dataset = dataset["train"]

processed_dataset = []
for instance in dataset:
    instruction = instance["instruction"]
    output = instance["output"]
    if 'python' in str.lower(instruction) or 'python' in str.lower(output):
        processed_dataset.append({
            "question": instruction,
            "answer": output
        })
print(len(processed_dataset))
with open("evol_instruct_code_filtered.json", "w") as file:
    json.dump(processed_dataset, file, indent=2)
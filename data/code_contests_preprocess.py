from datasets import load_dataset
import json

dataset = load_dataset("deepmind/code_contests")
dataset = dataset["train"]

processed_dataset = []
for instance in dataset:
    description = instance["description"]
    python_solution = None
    for lang, code in zip(instance["solutions"]["language"], instance["solutions"]["solution"]):
        if lang == 3:
            python_solution = code
            break
    if python_solution is None:
        continue
    processed_dataset.append({
        "question": description,
        "answer": python_solution
    })
print(len(processed_dataset))
with open("code_contests_filtered.json", "w") as file:
    json.dump(processed_dataset, file, indent=2)
from datasets import load_dataset
dataset = load_dataset("deepmind/code_contests")
dataset = dataset["train"]
def process_instance(instance):
    description = instance["description"]
    python_solution = None
    for lang, code in zip(instance["solutions"]["language"], instance["solutions"]["solution"]):
        if lang == 3:
            print
            python_solution = code
            break
    
    if python_solution is None:
        return
    return {
        "description": description,
        "python_solution": python_solution
    }
processed_dataset = list(map(process_instance, dataset))
print(processed_dataset[:10])
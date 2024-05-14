import json
files = ["Code-Generation-LLM-LoRA/data/code_contests_filtered.json", "Code-Generation-LLM-LoRA/data/code_contests_filtered.json"]
with open("Code-Generation-LLM-LoRA/data/lora_fine_tuning_data.json", "w") as file:
    for f in files:
        with open(f, "r") as f:
            for line in f:
                file.write(line)

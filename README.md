# Improving Llama-2-7B Code Generation for Python Code Using LoRA

Here's our full report:

[Final Report](./Final_Report.pdf)

# Installation and usage

## Installation

Install necessary libraries:

```bash
pip install -r requirements.txt
```

## Usage

### Inference

Run `inference.py` directly

### Fine-tuning model based on our filtered data

- Option #1 (Using Hugging Face PEFT library): Run `lora_peft.py`

- Option #2 (Using my hand-written version, may have bugs): Run `finetuning.py`


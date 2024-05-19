import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, B

from lora import lora, mark_only_lora_as_trainable


def train(model, train_dataset, val_dataset, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        learning_rate=3e-4,
        weight_decay=0.0,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda data: {"input_ids": torch.stack([f["input_ids"] for f in data]), "labels": torch.stack([f["labels"] for f in data])},
    )

    trainer.train()
    trainer.save_model(output_dir)


def main():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    data_dir = "Code-Generation-LLM-LoRA/data"

    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    with lora(r=8, alpha=16, dropout=0.05, enabled=True):
        model = LlamaForCausalLM.from_pretrained(model_name,)

    mark_only_lora_as_trainable(model)

    train_dataset = torch.load(os.path.join(data_dir, "train.pt"))
    val_dataset = torch.load(os.path.join(data_dir, "test.pt"))

    output_dir = "lora_checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    train(model, train_dataset, val_dataset, output_dir)


if __name__ == "__main__":
    main()
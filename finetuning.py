import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from lora import lora, mark_only_lora_as_trainable
from datasets import load_dataset

def generate_and_tokenize_prompt(example, tokenizer, max_input_length, max_target_length):
    tokenized_target = tokenizer(example["answer"],
                                 max_length=max_target_length - 1,
                                 truncation=True,
                                 add_special_tokens=False)
    tokenized_target["input_ids"] = tokenized_target["input_ids"] + [tokenizer.eos_token_id]
    tokenized_target["attention_mask"] = tokenized_target["attention_mask"] + [1]

    prompt = "Below is an instruction that describes a task.\nYou are an AI program expert. Your task is to solve programming problems from interviews and coding contests only using Python in detail.\n### Instruction:" + example["question"] + "\n### Response:\n"
    max_prompt_len = (max_input_length + max_target_length) - len(tokenized_target["input_ids"])
    model_inputs = tokenizer(prompt, max_length=max_prompt_len, padding="max_length", truncation=True)

    model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + tokenized_target["input_ids"]
    model_inputs["input_ids"] = model_inputs["input_ids"] + tokenized_target["input_ids"]
    model_inputs["attention_mask"] = model_inputs["attention_mask"] + tokenized_target["attention_mask"]
    return model_inputs

def train(model, train_dataset, val_dataset, output_dir, tokenizer):
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
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(output_dir)

def main():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    dataset_name = "Rabinovich/Code-Generation-LLM-LoRA"
    max_input_length = 512
    max_target_length = 256

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    with lora(r=8, alpha=16, dropout=0.05, enabled=True):
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    mark_only_lora_as_trainable(model)

    data = load_dataset(dataset_name)
    train_data = data["train"].shuffle().map(lambda x: generate_and_tokenize_prompt(x, tokenizer, max_input_length, max_target_length))
    val_data = data["test"].shuffle().map(lambda x: generate_and_tokenize_prompt(x, tokenizer, max_input_length, max_target_length))

    output_dir = "./model_7B_LoRA"

    train(model, train_data, val_data, output_dir, tokenizer, max_input_length, max_target_length)

if __name__ == "__main__":
    main()
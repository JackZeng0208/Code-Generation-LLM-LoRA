import torch
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import deepspeed
from accelerate.utils import DistributedType

def train(base_model, dataset_name, max_input_length, max_target_length, output_dir, resume_from_checkpoint=None):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def generate_and_tokenize_prompt(example):
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

    model = LlamaForCausalLM.from_pretrained(base_model)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print(model.print_trainable_parameters())
    data = load_dataset(dataset_name)
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)

    deepspeed_config = {
        "use_cache": False,
        "train_batch_size": 'auto',
        "train_micro_batch_size_per_gpu": 'auto',
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4
            }
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "scheduler": 
        {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 1e-4,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 100
            }
        }
    }

    training_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            bf16=True,
            num_train_epochs=4,
            learning_rate=1e-4,
            logging_steps=10,
            warmup_steps=100,
            output_dir=output_dir,
            deepspeed=deepspeed_config,
            resume_from_checkpoint=resume_from_checkpoint,
            save_strategy="no"
        )
        
    training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    trainer = Trainer(model=model, train_dataset=train_data, args=training_args, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))

    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    model.save_pretrained(output_dir)

if __name__ == "__main__":
    base_model = "meta-llama/Llama-2-7b-chat-hf"
    dataset_name = "Rabinovich/Code-Generation-LLM-LoRA"
    output_dir = "./model_7B_LoRA"
    max_input_length = 512
    max_target_length = 256

    # Set the path to the checkpoint directory if resuming from a checkpoint
    checkpoint_dir = "./checkpoint-25000"  # Replace with your actual checkpoint directory or set to None if not resuming

    train(base_model, dataset_name, max_input_length, max_target_length, output_dir, resume_from_checkpoint=checkpoint_dir)

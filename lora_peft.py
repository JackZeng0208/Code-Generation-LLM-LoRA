import torch
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import deepspeed

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
def train(base_model, dataset_name, max_input_length, max_target_length, output_dir):
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

        prompt = "Below is an instruction that describes a task.\nYou are an AI program assistant. Your task is to solve programming problems from interviews and coding contests only using Python.\n### Instruction:"+example["question"]+"\n### Response:\n"
        max_prompt_len = (max_input_length + max_target_length) - len(tokenized_target["input_ids"])
        model_inputs = tokenizer(prompt, max_length=max_prompt_len, padding="max_length", truncation=True)

        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + tokenized_target["input_ids"]
        model_inputs["input_ids"] = model_inputs["input_ids"] + tokenized_target["input_ids"]
        model_inputs["attention_mask"] = model_inputs["attention_mask"] + tokenized_target["attention_mask"]
        # print(model_inputs["input_ids"])
        return model_inputs


    model = LlamaForCausalLM.from_pretrained(base_model, quantization_config=bnb_config)
    # model.gradient_checkpointing_enable()
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
    
    data = load_dataset(dataset_name)
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    deepspeed_config = {
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
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
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        }
    }
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=5,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=10,
            output_dir=output_dir,
            deepspeed=deepspeed_config
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    base_model = "meta-llama/Llama-2-13b-chat-hf"
    dataset_name = "Rabinovich/Code-Generation-LLM-LoRA"
    output_dir = "Code-Generation-LLM-LoRA/model_13B"
    max_input_length = 512
    max_target_length = 256
    train(base_model, dataset_name, max_input_length, max_target_length, output_dir)
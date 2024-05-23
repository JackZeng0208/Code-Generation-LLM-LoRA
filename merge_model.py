from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from huggingface_hub import HfApi
import torch

lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16)
peft_model = PeftModel.from_pretrained(model, model_id="Code-Generation-LLM-LoRA/model_7B_LoRA", config=lora_config)
merged_model = peft_model.merge_and_unload()
output_dir = "Code-Generation-LLM-LoRA/combined_model"
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
api = HfApi()
api.upload_folder(
    folder_path=output_dir,
    repo_id="Rabinovich/Code-Generation-LLM-LoRA-Combined-Model"
)
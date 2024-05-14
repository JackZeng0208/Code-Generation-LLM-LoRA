from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftConfig, PeftModel
import os
import torch

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    # quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

peft_config = PeftConfig.from_pretrained("Code-Generation-LLM-LoRA/model/")
peft_model = PeftModel.from_pretrained(model, "Code-Generation-LLM-LoRA/model/", peft_config=peft_config)

peft_model.merge_and_unload()
# peft_model.save_pretrained("Code-Generation-LLM-LoRA/model/peft/")
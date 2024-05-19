import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
import torch

def generate_code(input_question, given_code, use_lora, max_new_tokens, top_p, top_k, temperature):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if use_lora:
        lora_path = "Code-Generation-LLM-LoRA/model"
        model = PeftModel.from_pretrained(model, model_id=lora_path, config=lora_config)

    text = [f"""###SYSTEM: Finish up remaining code based on INSTRUCTION and GIVEN CODE in Python:\n###INSTRUCTION: {input_question}\n###GIVEN CODE:\n{given_code}"""]

    inputs_ids = tokenizer(text, return_tensors="pt", padding=True).to('cuda:0')
    outputs = model.generate(
        **inputs_ids,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature
    )
    prompt_length = inputs_ids['input_ids'].shape[1]
    answer = tokenizer.decode(outputs[0][prompt_length:])
    return answer

iface = gr.Interface(
    fn=generate_code,
    inputs=[
        gr.Textbox(label="Input Question"),
        gr.Textbox(label="Given Code"),
        gr.Checkbox(label="Use LoRA Adapter"),
        gr.Slider(minimum=1, maximum=4096, step=1, value=512, label="Max New Tokens"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0, label="Top P"),
        gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Top K"),
        gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1, label="Temperature")
    ],
    outputs=gr.Code(language="python"),
    title="Code Generation with Llama-2-7b",
    description="Generate Python code based on the input question and given code."
)

iface.launch()
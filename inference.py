import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import torch

def generate_code(input_question, given_code, use_lora, prompt_choice, max_new_tokens, top_p, top_k, temperature):

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        torch_dtype=torch.bfloat16,
    ).to('cuda:0')
    model.eval()
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
        lora_path = "Code-Generation-LLM-LoRA/model_7B_LoRA"
        model = PeftModel.from_pretrained(model, model_id=lora_path, config=lora_config)

    prompt_1 = [f"""You are an high-level AI program expert. Your task is to solve programming problems from interviews and coding contests only using Python in detail. Given INSTRUCTION, finish up the RESPONSE part in detail based on given code:\n###INSTRUCTION: {input_question}\n###RESPONSE:\n# Given Code\n{given_code}\n"""]
    prompt_2 = [f"""You are an AI program assistant. Your task is to solve programming problems from interviews and coding contests only using Python. Given INSTRUCTION, finish up the RESPONSE part based on given code:\n###INSTRUCTION: {input_question}\n###RESPONSE:\n# Given Code\n{given_code}\n"""]

    if prompt_choice == "Prompt 1":
        text = prompt_1
    else:
        text = prompt_2

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
        gr.Radio(label="Prompt Choice", choices=["Prompt 1", "Prompt 2"]),
        gr.Slider(minimum=1, maximum=4096, step=1, value=512, label="Max New Tokens"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.5, label="Top P"),
        gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Top K"),
        gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1, label="Temperature")
    ],
    outputs=gr.Code(language="python"),
    title="Code Generation with Llama-2-7b",
    description="Generate Python code based on the input question and given code."
)

iface.launch(share=True)
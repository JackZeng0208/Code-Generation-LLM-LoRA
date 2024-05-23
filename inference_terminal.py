from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import torch

def generate_code(input_question, given_code, use_lora, max_new_tokens, top_p, top_k, temperature):
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

    text = [f"""Below is an instruction that describes a task. You are an AI program assistant. Your task is to solve programming problems from interviews and coding contests only using Python. You should provide the most precise and efficient code. Given INSTRUCTION, Solve the problem in detail based on GIVEN CODE:\n###INSTRUCTION: {input_question}\n###GIVEN CODE:\n{given_code}"""]

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

if __name__ == "__main__":
    while True:
        input_question = input("question: ") #"Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target."
        given_code = input("code: ")#"def twoSum(self, nums: List[int], target: int) -> List[int]:"
        use_lora = True
        max_new_tokens = 512
        top_p = 0.5
        top_k = 1
        temperature = 1.0

        generated_code = generate_code(input_question, given_code, use_lora, max_new_tokens, top_p, top_k, temperature)
        print("Generated Code:")
        print(generated_code)

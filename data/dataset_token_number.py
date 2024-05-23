from transformers import LlamaTokenizer
from datasets import load_dataset

total_token_number = 0
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
dataset = load_dataset("tatsu-lab/alpaca")
for example in dataset["train"]:
    total_token_number += len(tokenizer.encode(example['output'] + example['input']+example['instruction']+example['text']))
print(total_token_number)
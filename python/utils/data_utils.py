# utils/data_utils.py

from datasets import load_dataset
from transformers import GPT2Tokenizer

def load_and_preprocess_data():
    # Load the dataset
    dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset")
    
    # Initialize the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Set padding token to eos_token (GPT-2 doesn't have a pad_token by default)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the dataset using 'prompt' and 'chosen' as conversation pairs
    def tokenize_function(examples):
        # Concatenate 'prompt' and 'chosen' fields as input to the model
        inputs = [prompt + " " + chosen for prompt, chosen in zip(examples['prompt'], examples['chosen'])]
        return tokenizer(inputs, padding="max_length", truncation=True, max_length=1024)
    
    # Apply tokenization to the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    return tokenized_datasets

# test/test.py

import os
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import yaml

# Add the parent directory to the system path so we can import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

# Load configuration from config/config.yaml
with open("../config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained(config['test']['model_path'])
tokenizer = GPT2Tokenizer.from_pretrained(config['test']['model_path'])

# Put the model in evaluation mode
model.eval()

def generate_response(input_text):
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Generate response from the model
    with torch.no_grad():
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=1024, num_return_sequences=1)
    
    # Decode the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("Chatbot ready! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        response = generate_response(user_input)
        print(f"Bot: {response}")

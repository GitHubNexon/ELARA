import os
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import yaml

# Add the parent directory to the system path so we can import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from data_utils import load_and_preprocess_data  # Now we can import from utils.data_utils

# Load configuration from config/config.yaml
with open("../config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Load and preprocess dataset
datasets = load_and_preprocess_data()

# Manually split the dataset into train and validation sets
train_dataset = datasets['train']
eval_dataset = train_dataset.train_test_split(test_size=0.1)['test']  # Split 90% for training, 10% for validation

# Initialize the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

learning_rate = float(config['train']['learning_rate'])

# Set training arguments
training_args = TrainingArguments(
    output_dir=config['train']['output_dir'],
    overwrite_output_dir=True,
    num_train_epochs=config['train']['epochs'],
    per_device_train_batch_size=config['train']['batch_size'],
    per_device_eval_batch_size=config['train']['batch_size'],
    logging_dir=config['train']['logging_dir'],
    save_steps=config['train']['save_steps'],
    evaluation_strategy="steps",
    logging_steps=100,
    learning_rate=learning_rate 
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)


# Train the model
trainer.train()

# Save the model
model.save_pretrained(config['train']['output_dir'])
tokenizer.save_pretrained(config['train']['output_dir'])

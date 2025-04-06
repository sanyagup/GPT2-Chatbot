from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch

# Function to load and label combined data (background + medical)
def load_and_label_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    labeled_data = "<background> " + "".join(data).replace('\n', ' <background> ')
    return labeled_data

# Tokenize the data with attention masks
def tokenize_data(data, tokenizer, block_size=128):
    encodings = tokenizer(data, return_tensors='pt', truncation=True, padding='max_length', max_length=block_size)
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Add special pad token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load and label the combined data
combined_data = load_and_label_data('combined_medical_data.txt')
tokenized_data = tokenize_data(combined_data, tokenizer)

# Create a dataset from the tokenized data
combined_dataset = Dataset.from_dict(tokenized_data)

# Load the GPT-2 model and resize token embeddings
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision training
)

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8  # Optional, but can speed up training on certain hardware
)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=combined_dataset,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./fine-tuned-gpt2-medical-enhanced')
tokenizer.save_pretrained('./fine-tuned-gpt2-medical-enhanced')

# Function to generate a response
def generate_response(prompt, model, tokenizer, max_length=100):
    encodings = tokenizer(prompt, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Query the model with a medical prompt
prompt = "<medical> What are the common symptoms of patients with a cholesterol level of 250?"
response = generate_response(prompt, model, tokenizer)
print(response)

def load_and_label_data(conversation_file, medical_file, output_file):
    with open(conversation_file, 'r') as file:
        conversational_data = file.read()
    with open(medical_file, 'r') as file:
        medical_data = file.read()
    
    labeled_conversational_data = "<conversation> " + conversational_data.replace('\n', ' <conversation> ')
    labeled_medical_data = "<medical> " + medical_data.replace('\n', ' <medical> ')
    combined_data = labeled_conversational_data + '\n' + labeled_medical_data
    
    with open(output_file, 'w') as file:
        file.write(combined_data)



from arc_loader import ArcAgiDatasetLoader
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch

# Your existing code
arc_loader = ArcAgiDatasetLoader()

# Load from saved_annotations.json
with open("saved_annotations.jsonl", "r", encoding="utf-8") as f:
    saved_annotations = [f.strip() for f in f if f.strip()]
hashes = {}
for entry in saved_annotations:
    data = json.loads(entry)
    hashes[data["task_id"]] = data["nlp_program"]
    hashes[data["task_id"]] = [hashes[data["task_id"]]] + data["variations"]

train_hashes = list(hashes.keys())[:45]
val_hashes = list(hashes.keys())[45:50]

# Now for each hash we fetch the grid pairs and we match the input output pair with the nlp programs
# So if we have 5 input output pairs and 3 nlp programs, we create 15 training examples
train_data = []
for h in train_hashes:
    pairs = arc_loader.get_all_pairs(h)
    for p in pairs:
        for nlp in hashes[h]:
            train_data.append({
                "task_id": h,
                "input": str(p[0]),
                "output": str(p[1]),
                "nlp_program": nlp
            })

val_data = []
for h in val_hashes:
    pairs = arc_loader.get_all_pairs(h)
    for p in pairs:
        for nlp in hashes[h]:
            val_data.append({
                "task_id": h,
                "input": str(p[0]),
                "output": str(p[1]),
                "nlp_program": nlp
            })
print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(val_data)}")

# ============= NEW TRAINING CODE =============

# Load Qwen 3B model and tokenizer
model_name = "Qwen/Qwen2.5-3B"  # or "Qwen/Qwen2.5-3B-Instruct" for instruction-tuned version
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use fp16 for memory efficiency
    device_map="auto"
)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Format data into prompts
def format_prompt(example):
    prompt = f"""Task: Transform the input grid according to the program description.

Program: {example['nlp_program']}
Input Grid: {example['input']}
Output Grid: {example['output']}"""
    return prompt

# Prepare datasets
def prepare_dataset(data):
    texts = [format_prompt(item) for item in data]
    return Dataset.from_dict({"text": texts})

train_dataset = prepare_dataset(train_data)
val_dataset = prepare_dataset(val_data)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=2048  # Adjust based on your grid sizes
    )

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./qwen-3b-arc-sft",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Adjust based on your GPU memory
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch size = 2 * 8 = 16
    warmup_steps=100,
    learning_rate=2e-5,
    fp16=True,  # Mixed precision training
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",  # Set to "wandb" or "tensorboard" if you want logging
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, not masked LM
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
print("Starting training...")
trainer.train()

# Save final model
trainer.save_model("./qwen-3b-arc-final")
tokenizer.save_pretrained("./qwen-3b-arc-final")

print("Training completed!")

# Optional: Test inference
def generate_output(input_grid, nlp_program):
    prompt = f"""Task: Transform the input grid according to the program description.

Program: {nlp_program}
Input Grid: {input_grid}
Output Grid:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test on a validation example
if val_data:
    test_example = val_data[0]
    result = generate_output(test_example["input"], test_example["nlp_program"])
    print(f"\nTest generation:\n{result}")
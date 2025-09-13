import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
# --- MODIFIED: Define separate files for training and validation ---
TRAIN_DATA_FILE = "train.jsonl"      # <-- MODIFIED: Path to your training data
EVAL_DATA_FILE = "validation.jsonl"  # <-- NEW: Path to your validation data
TRAINED_MODEL_PATH = "./greymatter-confluence-lora-12-09-25"
# ---------------------------
# 1. Model & Tokenizer
# ---------------------------
model_id = BASE_MODEL_ID

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # avoid padding issues

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0}   # force GPU 0
)

# ---------------------------
# 2. Add LoRA Adapters
# ---------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)


# ---------------------------
# 3. Dataset
# ---------------------------
# --- MODIFIED: Load both training and validation sets ---
data_files = {"train": TRAIN_DATA_FILE, "validation": EVAL_DATA_FILE} # <-- MODIFIED
dataset = load_dataset("json", data_files=data_files)                 # <-- MODIFIED

def format_prompt(example):
    # The official Mistral Instruct template
    if example.get("input"):
        text = f"<s>[INST] {example['instruction']}\n{example['input']} [/INST] {example['output']}</s>"
    else:
        text = f"<s>[INST] {example['instruction']} [/INST] {example['output']}</s>"
    return {"text": text}

# Apply the formatting to both train and validation splits
formatted_dataset = dataset.map(format_prompt)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

# Tokenize both datasets
tokenized_datasets = formatted_dataset.map(
    tokenize_function,
    batched=True, # Process in batches for efficiency
    remove_columns=dataset["train"].column_names + ["text"] # Clean up old columns
)

# Assign to final variables
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"] # <-- NEW: Create the evaluation dataset

# ---------------------------
# 4. Training Setup
# ---------------------------
with open(TRAIN_DATA_FILE, "r") as f:
    num_examples = sum(1 for _ in f)

print(f"Found {num_examples} training examples")

per_device_train_batch_size = 2
gradient_accumulation_steps = 16
epochs = 5

steps_per_epoch = (num_examples // (per_device_train_batch_size * gradient_accumulation_steps)) + 1
max_steps = steps_per_epoch * epochs

print(f"Training for {epochs} epochs = {max_steps} steps total")

training_args = TrainingArguments(
    output_dir="./gcm-lora-out",
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=10,
    max_steps=max_steps,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    # --- NEW: Add evaluation arguments ---
    evaluation_strategy="epoch",          # <-- NEW: Run evaluation at the end of each epoch
    per_device_eval_batch_size=per_device_train_batch_size, # <-- NEW: Use same batch size for eval
    report_to="tensorboard"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # <-- NEW: Pass the validation dataset to the Trainer
    data_collator=data_collator
)

# ---------------------------
# 5. Train
# ---------------------------
trainer.train()

# ---------------------------
# 6. Save LoRA Adapter
# ---------------------------
model.save_pretrained(TRAINED_MODEL_PATH)
tokenizer.save_pretrained(TRAINED_MODEL_PATH)

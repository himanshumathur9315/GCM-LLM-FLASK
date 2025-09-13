import torch
import transformers
import inspect
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
from pkg_resources import parse_version

# --- Add a version check for the transformers library ---
print(f"Using transformers version: {transformers.__version__}")


BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
# --- MODIFIED: Define separate files for training and validation ---
TRAIN_DATA_FILE = "training.jsonl"
EVAL_DATA_FILE = "validation.jsonl"
TRAINED_MODEL_PATH = "./simulator-confluence-lora-13-09-25"
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
data_files = {"train": TRAIN_DATA_FILE, "validation": EVAL_DATA_FILE}
dataset = load_dataset("json", data_files=data_files)

def format_prompt(example):
    # The official Mistral Instruct template
    if example.get("input"):
        text = f"<s>[INST] {example['instruction']}\n{example['input']} [/INST] {example['output']}</s>"
    else:
        text = f"<s>[INST] {example['instruction']} [/INST] {example['output']}</s>"
    return {"text": text}

# Apply the formatting to both train and validation splits
formatted_dataset = dataset.map(format_prompt)

# --- MODIFIED: Tokenize both splits using the working script's method ---
def tokenize_text(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

train_dataset = formatted_dataset["train"].map(
    tokenize_text,
    remove_columns=formatted_dataset["train"].column_names
)
eval_dataset = formatted_dataset["validation"].map(
    tokenize_text,
    remove_columns=formatted_dataset["validation"].column_names
)

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

# --- DEFINITIVE FIX: Use environment-aware arguments ---
common_args = {
    "output_dir": "./gcm-lora-out",
    "per_device_train_batch_size": per_device_train_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "warmup_steps": 10,
    "max_steps": max_steps,
    "learning_rate": 2e-4,
    "fp16": True,
    "logging_steps": 10,
    "save_steps": steps_per_epoch,
    "save_total_limit": 2,
    "report_to": "tensorboard"
}

# --- DEFINITIVE FIX: Dynamically select the correct argument name based on your diagnostic test ---
training_args_signature = inspect.signature(TrainingArguments).parameters
if "eval_strategy" in training_args_signature:
    print("Using 'eval_strategy' for evaluation strategy.")
    common_args["eval_strategy"] = "epoch"
else:
    print("Using 'evaluation_strategy' for evaluation strategy.")
    common_args["evaluation_strategy"] = "epoch"

# Add the per_device_eval_batch_size argument, which is consistent across versions
common_args["per_device_eval_batch_size"] = per_device_train_batch_size

training_args = TrainingArguments(**common_args)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- MODIFIED: Add eval_dataset to the Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # <-- ADDED
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

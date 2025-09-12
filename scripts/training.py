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
DATA_SET_FILE = "merged.jsonl"
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
# 3. Dataset (Corrected)
# ---------------------------
# dataset = load_dataset("json", data_files=DATA_SET_FILE)

# def format_prompt(example):
#     # assumes keys: instruction, input, output
#     if example.get("input"):
#         prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}\nAnswer:"
#     else:
#         prompt = f"Instruction: {example['instruction']}\nAnswer:"
#     return {
#         "input_ids": tokenizer(prompt, truncation=True, padding="max_length", max_length=512).input_ids,
#         "labels": tokenizer(example["output"], truncation=True, padding="max_length", max_length=512).input_ids
#     }

# train_dataset = dataset["train"].map(format_prompt, remove_columns=dataset["train"].column_names)

dataset = load_dataset("json", data_files=DATA_SET_FILE)

# This function now creates a single formatted text string
def format_prompt(example):
    # The official Mistral Instruct template
    if example.get("input"):
        # This handles instructions that have both an "instruction" and "input" field
        text = f"<s>[INST] {example['instruction']}\n{example['input']} [/INST] {example['output']}</s>"
    else:
        # This handles instructions that only have an "instruction" field
        text = f"<s>[INST] {example['instruction']} [/INST] {example['output']}</s>"
    return {"text": text}

# Apply the formatting
formatted_dataset = dataset.map(format_prompt)

# Tokenize the formatted text
# The Trainer will automatically use the 'input_ids' and create 'labels'
train_dataset = formatted_dataset["train"].map(
    lambda x: tokenizer(
        x["text"],
        truncation=True,
        padding="max_length",
        max_length=512  # You can adjust this length
    ),
    remove_columns=dataset["train"].column_names + ["text"] # Clean up old columns
)


# The rest of your training script (Section 4, 5, 6) can remain the same.
# Just replace your old dataset section with this one and retrain the model.



# ---------------------------
# 4. Training Setup
# ---------------------------
jsonl_path=DATA_SET_FILE
# Parameters
with open(jsonl_path, "r") as f:
    num_examples = sum(1 for _ in f)

print(f"Found {num_examples} training examples")

# Training parameters
per_device_train_batch_size = 2
gradient_accumulation_steps = 16   # effective batch size = 16
epochs = 5                        # train for 5 epochs

# Compute steps per epoch
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
    save_steps=50,         # save more frequently since dataset is small
    save_total_limit=2,
    report_to="tensorboard"
)



# training_args = TrainingArguments(
#     output_dir="./gcm-lora-out",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=8,
#     warmup_steps=10,
#     max_steps=200,   # small run for test
#     learning_rate=2e-4,
#     fp16=True,
#     logging_steps=10,
#     save_steps=100,
#     save_total_limit=2,
#     report_to="none"
# )

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
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

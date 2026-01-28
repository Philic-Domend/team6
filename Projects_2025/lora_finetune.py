import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# Paths and configurations
model_name_or_path = "/home/kangshuo/.cache/modelscope/hub/models/Qwen/Qwen2-1.5B-Instruct"
data_dir = "gsm8k"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Task type for causal language modeling
    inference_mode=False,
    r=8,  # Rank of the LoRA matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1  # Dropout probability
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

# Load dataset
dataset = load_dataset("json", data_files={
    "train": os.path.join(data_dir, "train.jsonl"),
    "test": os.path.join(data_dir, "test.jsonl")
})

# Reduce max_length in preprocessing
def preprocess_function(examples):
    """
    Preprocess the dataset to format it for training.
    """
    inputs = examples["question"]
    targets = examples["answer"]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")

    # Tokenize the labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_steps=500,  # Evaluate every 500 steps
    learning_rate=5e-5,
    per_device_train_batch_size=1,  # Reduce batch size to fit GPU memory
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,  # Enable mixed precision training for faster training
    push_to_hub=False
)

# Add memory management
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # Optimize GPU performance

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

# Start training
if __name__ == "__main__":
    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("./results")

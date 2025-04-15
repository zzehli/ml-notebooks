#!/usr/bin/env python
# coding: utf-8
import subprocess
import sys
import os
subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "transformers", "evaluate", "accelerate", "sentencepiece", "huggingface_hub"])
# Install Git LFS
subprocess.run(
    "curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash",
    shell=True, check=True
)

subprocess.run(
    "apt-get install -y git-lfs",
    shell=True, check=True
)

from transformers import LlamaTokenizer
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import DataCollatorForLanguageModeling, LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

from huggingface_hub import login

login(token=os.getenv("TOKEN"))


train_data = load_dataset(f"roneneldan/TinyStories", split="train")
validation_data = load_dataset(f"roneneldan/TinyStories", split="validation")
raw_datasets = DatasetDict(
    {
        "train": train_data,
        "valid": validation_data,
    }
)
print(raw_datasets)
context_length = 512
tokenizer =  LlamaTokenizer.from_pretrained("facebook/MobileLLM-125M")
tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>"})

def tokenize(elements):
    # Concatenate all texts in the batch with EOS token between them
    concatenated_text = tokenizer.eos_token.join(elements["text"])
    
    # Tokenize the concatenated text
    outputs = tokenizer(
        concatenated_text,
        truncation=False,  # Don't truncate yet, we'll handle chunking manually
        return_tensors="pt",
        verbose=False
    )
    
    input_ids = outputs["input_ids"][0]  # Extract the token IDs
    
    # Create chunks of size context_length
    total_length = input_ids.size(0)
    chunks = []
    
    for i in range(0, total_length, context_length):
        chunk = input_ids[i:i + context_length].tolist()
        # Only keep chunks that are full or close to full (e.g., at least 50% of context_length)
        if len(chunk) >= 0.5 * context_length:
            chunks.append(chunk)
    
    return {"input_ids": chunks}

tokenized_datasets = raw_datasets.map(
    tokenize, 
    batched=True, 
    # batch_size=32,  # Adjust based on your memory constraints
    num_proc=4, 
    remove_columns= raw_datasets["train"].column_names
)
print(tokenized_datasets)
tokenized_datasets.save_to_disk("tokenized_dataset")
print("============Tokenized dataset saved to disk.")
# tokenized_datasets = load_from_disk("tokenized_dataset")



tokenizer.decode(tokenized_datasets['train'][0]['input_ids'])
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

configuration = LlamaConfig(
    hidden_size=256,
    intermediate_size=512,
    max_position_embeddings=context_length,
    num_attention_heads=8,
    num_hidden_layers=30,
    tie_word_embeddings=True,
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id,
    )

model = LlamaForCausalLM(configuration)

model_size = sum(t.numel() for t in model.parameters())
print(f"GPT size: {model_size/1000**2:.1f}M parameters")


print("BOS token:", tokenizer.bos_token, "ID:", tokenizer.bos_token_id)
print("EOS token:", tokenizer.eos_token, "ID:", tokenizer.eos_token_id)
print("PAD token:", tokenizer.pad_token, "ID:", tokenizer.pad_token_id)

print("Model config BOS ID:", model.config.bos_token_id)
print("Model config EOS ID:", model.config.eos_token_id)
print("Model config PAD ID:", model.config.pad_token_id)

args = TrainingArguments(
    output_dir="llama-fin",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="steps",
    eval_steps=5_000,
    logging_steps=3_000,
    gradient_accumulation_steps=1,
    weight_decay=0.1,
    warmup_steps=1000,
    lr_scheduler_type="cosine",
    auto_find_batch_size=True,
    learning_rate=3e-4,
    num_train_epochs=2,
    save_strategy="best",
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    bf16=True,
    torch_compile=True
)

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

trainer.train()
trainer.push_to_hub()

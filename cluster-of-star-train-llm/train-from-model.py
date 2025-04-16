#!/usr/bin/env python
# coding: utf-8
# import subprocess
# import sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "transformers", "evaluate", "accelerate", "sentencepiece", "huggingface_hub", "tensorboard"])
# # Install Git LFS
# subprocess.run(
#     "curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash",
#     shell=True, check=True
# )

# subprocess.run(
#     "apt-get install -y git-lfs",
#     shell=True, check=True
# )

from transformers import LlamaTokenizer, LlamaForCausalLM, DataCollatorForLanguageModeling, LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, load_from_disk

tokenizer =  LlamaTokenizer.from_pretrained("Jae-star/llama-fin")
model = LlamaForCausalLM.from_pretrained("Jae-star/llama-fin")
# tokenized_datasets = load_from_disk("tokenized_dataset")
tokenized_datasets = load_dataset(f"Jae-star/tinystories_llama_nogroup")

tokenized_datasets = tokenized_datasets.remove_columns("attention_mask")

# train_datasets = tokenized_datasets["train"].shard(num_shards=6, index=0)
valid_datasets = tokenized_datasets["valid"].shard(num_shards=6, index=0)
valid_datasets = valid_datasets.filter(lambda example: len(example['input_ids'])<512)

# print(train_datasets)

# train_datasets = train_datasets.filter(lambda example: len(example['input_ids'])<512)
# print(train_datasets)
# train_datasets.save_to_disk("tokenized_dataset")
train_datasets = load_from_disk("tokenized_dataset")

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

model_size = sum(t.numel() for t in model.parameters())
print(f"GPT size: {model_size/1000**2:.1f}M parameters")

# print("==============is bf16 supported?")
# print(torch.cuda.is_bf16_supported())
args = TrainingArguments(
    output_dir="llama-fin-re",
    log_level="info",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=100,
    gradient_accumulation_steps=12,
    weight_decay=0.1,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    auto_find_batch_size=True,
    learning_rate=5e-5,
    num_train_epochs=1,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    # bf16=True,
    torch_compile=True,
    report_to="tensorboard",
    eval_accumulation_steps=200
)

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=train_datasets,
    eval_dataset=valid_datasets,
)

trainer.train()
trainer.push_to_hub()
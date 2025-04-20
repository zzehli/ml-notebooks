* align tokenizer with model
* check tokenizer output: decode to check the output at every step
    * overflow: 
    * truncation
    * batching: if input is batched, then tokenizer should handle that
    * collating: output of a batch should be the same length
* batching: if context window is big, concatenate content
* decoding strategies: https://huggingface.co/docs/transformers/en/generation_strategies#contrastive-search
* process dataset: https://huggingface.co/docs/datasets/en/process
* no batching
```
def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for input_ids in outputs["input_ids"]:
        input_batch.append(input_ids)
    return {"input_ids": input_batch}

tokenized_datasets = test_ds.map(
    tokenize, batched=True, num_proc = 4, remove_columns=raw_datasets["train"].column_names
)
tokenized_datasets
```
* batching
```
def tokenize_cd(elements):
    # Concatenate all texts in the batch with EOS token between them
    concatenated_text = tokenizer.eos_token.join(elements["text"])
    
    # Tokenize the concatenated text
    outputs = tokenizer(
        concatenated_text,
        truncation=False,  # Don't truncate yet, we'll handle chunking manually
        return_tensors="pt",
    )
    
    input_ids = outputs["input_ids"][0]  # Extract the token IDs
    
    # Create chunks of size context_length
    total_length = input_ids.size(0)
    chunks = []
    
    for i in range(0, total_length, context_length):
        chunk = input_ids[i:i + context_length].tolist()
        # Only keep chunks that are full or close to full (e.g., at least 80% of context_length)
        if len(chunk) >= 0.5 * context_length:
          chunks.append(chunk)
    
    return {"input_ids": chunks}

tokenized_cd = test_ds.map(
    tokenize_cd, 
    batched=True, 
    # batch_size=32,  # Adjust based on your memory constraints
    num_proc=4, 
    remove_columns=test_ds.column_names
)
print(tokenized_cd)
for i in tokenized_cd["input_ids"]:
  print(len(i))
print('**')
```
* Q: Let's say my batch size * length of each sample will be far more than my context length. I run through my dataset, it turns out whether i set 20 or 40, the resulting dataset has a similar size, why?
A: When your concatenated text (batch_size * avg_length) is much larger than your context_length, the number of chunks you get is primarily determined by the total number of tokens in your dataset, not by how you group them initially.
For example:

Let's say each sample averages 200 tokens
Your context_length is 2048 tokens
Your total dataset has 100,000 samples (20 million tokens total)

If you process with batch_size=20:

Each concatenated sequence is ~4,000 tokens (20 * 200)
Each concatenated sequence yields ~2 full chunks
Processing all 100,000 samples gives you ~10,000 chunks

If you process with batch_size=40:

Each concatenated sequence is ~8,000 tokens (40 * 200)
Each concatenated sequence yields ~4 full chunks
Processing all 100,000 samples still gives you ~10,000 chunks

In both cases, you're processing all 20 million tokens and chunking them into 2048-token pieces. The initial grouping doesn't matter much because you're re-chunking everything anyway.
* run time batch size
> The auto_find_batch_size argument is an optional argument which can be used in addition to the per_device_train_batch_size argument.
* batch size in tokenization vs. training are different
* some parameters: hidden size, attention head, layer/hidden layer, intermediate size, position embedding
    * https://www.galileo.ai/blog/llm-parameters-model-evaluation
* mobileLLM:
> The initial learning rate is set
> to 2e-3 and follows a cosine learning-rate decay strategy.
* sequence packing (concatenate sequences and adjust masks): https://huggingface.co/blog/sirluk/llm-sequence-packing
* speed up training 
* inference on GPU
* common commands
```
python3 checkpoints-train.py > output.log 2>&1 &
tail -f output.log
ps aux | grep python
```
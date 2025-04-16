from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import pandas as pd
df = pd.read_csv("final_evaluation_prompts.csv")  # Assumes there's a 'prompt' column

tokenizer =  LlamaTokenizer.from_pretrained("Jae-star/llama-fin-re", padding_side='left')
model = LlamaForCausalLM.from_pretrained("Jae-star/llama-fin-re")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
prompts = df["prompt"].tolist()
# prompts = prompts = ["Fish has a big dream.",
#                      "Fish has a big dream.",
#                      "Fish has a big dream.",
#            "Apple tree is close by.",
#            "Ann wants to go to university",
#            "He left early in the morning befor the sunrise. But he hasn't come back after a whole day",
#            "Clearly there is something sketchy going on in the school."]
# print(prompts)
# print(type(prompts))

input_ids = tokenizer(prompts, return_tensors="pt", padding=True)
input_ids.to(device)
gen_tokens = model.generate(
    **input_ids,
    max_length=300,
    do_sample=True,
    num_beams=2,
    # top_p=0.9,
    # repetition_penalty=1.2,
)
prompt_length = input_ids['input_ids'].shape[1]

gen_text = tokenizer.batch_decode(gen_tokens[:,prompt_length:])

df["completion"]=gen_text
df.to_csv("final_eval_complete_re_bem_mulp.csv", index=False)
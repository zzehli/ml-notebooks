import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient, login

load_dotenv()

login(token=os.getenv("HUGGINGFACE_API_KEY"))


def chat(messages: list[str]) -> str:
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    client = InferenceClient(
        api_key=os.getenv("HUGGINGFACE_API_KEY"),
    )

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            # max_tokens=200,
        )
    except Exception as e:
        print(f"Error: {e}")
        return
    return response.choices[0].message.content

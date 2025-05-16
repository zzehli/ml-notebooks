import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient, login
from transformers import AutoTokenizer
from utils import function_to_schema

load_dotenv()

login(token=os.getenv("HUGGINGFACE_API_KEY"))


def get_name(name: str) -> str:
    """Get the name of the anyone the user asks."""
    return "John Doe"


tools = [function_to_schema(get_name)]


def main():
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    client = InferenceClient(
        api_key=os.getenv("HUGGINGFACE_API_KEY"),
    )
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the name tool to get the name of the person user asks.",
        },
        {
            "role": "user",
            "content": "Who is the person over there?",
        },
    ]

    input = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    # observe the templated input before and after adding tools (adding tools apply additional instructions to the system prompt)
    # see details here: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/main/chat_template.json
    print(input)

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=tools,
            max_tokens=200,
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    # again, try make requests with and without tools.
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()

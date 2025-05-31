import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient, login

load_dotenv()

login(token=os.getenv("HUGGINGFACE_API_KEY"))


def main():
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    client = InferenceClient(
        api_key=os.getenv("HUGGINGFACE_API_KEY"),
    )
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the user's questions in a concise manner.",
        },
        {
            "role": "user",
            "content": "Who are Jack, Sam, and Mary?",
        },
    ]

    while True:
        input_text = input("User: ")
        if input_text.lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": input_text})
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=200,
            )
        except Exception as e:
            print(f"Error: {e}")
            return

        print(response.choices[0].message.content)
        messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
    for m in messages:
        print(m)


if __name__ == "__main__":
    main()

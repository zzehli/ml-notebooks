import os

from client import chat
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

login(token=os.getenv("HUGGINGFACE_API_KEY"))


def main():
    messages = []

    while True:
        input_text = input("User: ")
        if input_text.lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": input_text})
        response = chat(messages)

        print(response)
        messages.append({"role": "assistant", "content": response})
    for m in messages:
        print(m)


if __name__ == "__main__":
    main()

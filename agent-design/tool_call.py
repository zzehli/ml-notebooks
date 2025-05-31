import os
import subprocess

from dotenv import load_dotenv
from huggingface_hub import InferenceClient, login
from pydantic import BaseModel
from transformers import AutoTokenizer
from utils import function_to_schema

load_dotenv()

login(token=os.getenv("HUGGINGFACE_API_KEY"))


class Person(BaseModel):
    name: str


class Directory(BaseModel):
    dir: str


def get_name(name: str) -> str:
    """Get the name of the anyone the user asks."""
    return "John Doe"


def browse_file(dir: str) -> str:
    """Browse files in the given directory.

    Args:
        dir (str): The directory to browse.

    Returns:
        A list of files in the directory.
    """
    if not os.path.exists(dir):
        return f"Directory {dir} does not exist."
    try:
        output = subprocess.check_output(["ls", dir], text=True)
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


tools = [function_to_schema(get_name), function_to_schema(browse_file)]


def main():
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    client = InferenceClient(
        api_key=os.getenv("HUGGINGFACE_API_KEY"),
    )
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the user's questions in a concise manner. Use appropriate tools when necessary.",
        },
        {
            "role": "user",
            # "content": "Who are Jack, Sam, and Mary?",
            "content": "What are the files in the current directory.",
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
            tool_choice="required",
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    # again, try make requests with and without tools.
    print("Response: ", response)
    # we make assumptions about the shape of the response, so we check if it has tool calls
    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            tool_name = tool_call.function.name
            args = tool_call.function.arguments
            print(f"Tool call: {tool_name} with args {args}")
            # print(f"Parsed args: {args}")
            if tool_name == "get_name":
                p = Person.model_validate_json(args)
                result = get_name(p.name)
            elif tool_name == "browse_file":
                d = Directory.model_validate_json(args)
                result = browse_file(d.dir)
            else:
                result = "Unknown tool"
            print(f"Tool result: {result}")


if __name__ == "__main__":
    main()

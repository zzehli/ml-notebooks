import inspect
import os
import subprocess

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()


class Tool:
    @staticmethod
    def get_name(name: str) -> str:
        """Get the name of the anyone the user asks."""
        return "John Doe"

    @staticmethod
    def browse_file(dir: str) -> str:
        """Browse files in the given directory."""
        if not os.path.exists(dir):
            return f"Directory {dir} does not exist."
        try:
            output = subprocess.check_output(["ls", dir], text=True)
            return output
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")


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


def function_to_schema(func) -> dict:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

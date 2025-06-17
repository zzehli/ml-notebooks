import json
import os
import traceback

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from litellm import completion
from openai import OpenAI
from utils import Tool, function_to_schema

load_dotenv()

default_system_message = "you are a helpful assistant. Answer the user's questions in a concise manner. Use appropriate tools when necessary."


class Agent:
    def __init__(
        self, tools=None, system_message=default_system_message, client="github"
    ):
        self.messages = [{"role": "system", "content": system_message}]
        self.tools = tools if tools is not None else []
        self.client_type = client
        # switched to openai since it has better tool calling
        match client:
            case "github":
                self.client = OpenAI(
                    api_key=os.environ.get("GITHUB_OPENAI_API_KEY"),
                    base_url="https://models.github.ai/inference",
                )
                self.model = "openai/gpt-4.1"
            case "google":
                self.model = "gemini/gemini-2.0-flash"
            case "huggingface":
                self.client = InferenceClient(api_key=os.getenv("HUGGINGFACE_API_KEY"))
                self.model = "meta-llama/Llama-3.3-70B-Instruct"
            case "qwen":
                self.client = InferenceClient(
                    api_key=os.getenv("HUGGINGFACE_API_KEY"),
                )
                self.model = "Qwen/Qwen3-14B"
            case _:
                raise ValueError(f"Invalid client: {client}")

    def _call(self, input, stop=None):
        match self.client_type:
            case "google":
                return completion(
                    model=self.model,
                    messages=input,
                    tools=[function_to_schema(x) for x in self.tools],
                    stop=stop,
                    tool_choice="auto",
                )
            case _:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=input,
                    tools=[function_to_schema(x) for x in self.tools],
                    tool_choice="auto",
                    stop=stop,
                )

    def _execute_tool(self, tool_calls):
        tool_results = []
        for tool_call in tool_calls:
            # a more generic tool execution function, more prune to errors, only works for string parameters, see react_agent for type conversion
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"Executing tool: {tool_name} with args: {args}")
            tool_dict = {tool.__name__: tool for tool in self.tools}
            res = tool_dict[tool_name](**args)
            print(f"Tool result: {res}")
            tool_results.append(
                {
                    "tool_name": tool_name,
                    "args": tool_call.function.arguments,
                    "result": res,
                }
            )
        return tool_results

    def run(self):
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Agent terminated.")
                    break
                self.messages.append({"role": "user", "content": user_input})
                response = self._call(self.messages)
                if response.choices[0].message.tool_calls:
                    tool_results = self._execute_tool(
                        response.choices[0].message.tool_calls
                    )
                    tool_results_str = "\n".join(
                        f"Tool '{result['tool_name']}' returned: {result['result']}"
                        for result in tool_results
                    )
                    self.messages.append({"role": "user", "content": tool_results_str})
                else:
                    assistant_message = response.choices[0].message.content
                    formatted_message = {
                        "role": "assistant",
                        "content": assistant_message,
                    }
                    print(f"Assistant: {assistant_message}")
                    self.messages.append(formatted_message)

            except Exception as e:
                print(f"Error: {e}")
                print(traceback.format_exc())
                return "An error occurred while processing your request."


if __name__ == "__main__":
    agent = Agent(tools=[Tool.browse_file], client="google")
    agent.run()

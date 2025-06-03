import json
import os
import traceback

from dotenv import load_dotenv
from openai import OpenAI
from utils import Tool, function_to_schema

load_dotenv()


class Agent:
    def __init__(self, tools=None):
        self.system_message = "you are a helpful assistant. Answer the user's questions in a concise manner. Use appropriate tools when necessary."
        self.messages = [{"role": "system", "content": self.system_message}]
        self.tools = tools if tools is not None else []
        # switched to openai since it has better tool calling
        self.client = OpenAI(
            api_key=os.environ.get("GITHUB_OPENAI_API_KEY"),
            base_url="https://models.github.ai/inference",
        )

    def _call(self, input, stop=None):
        return self.client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=input,
            tools=[function_to_schema(x) for x in self.tools],
            tool_choice="auto",
            stop=stop,
        )

    def _execute_tool(self, tool_calls):
        tool_results = []
        for tool_call in tool_calls:
            # a more generic tool execution function, more prune to errors
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"Executing tool: {tool_name} with args: {args}")
            print(f"all tools: {self.tools}")
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
    agent = Agent(tools=[Tool.browse_file])
    # agent = Agent()
    agent.run()

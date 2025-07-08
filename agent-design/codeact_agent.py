import json
from typing import Any

from agent import Agent
from pydantic import BaseModel
from rich import print as rprint
from rich.syntax import Syntax
from rich.text import Text
from smolagents.local_python_executor import LocalPythonExecutor
from smolagents.tools import Tool
from utils import print_step_line

# we use a elaborate setup to use smolagent's isolated python execution environment, that's why we wrap tools as smolagent's Tool class. This allows us to send tools to the custom python environment.

prompt = """
You are an expert assistant who can solve any task using python code. To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'thought:' attribute, you should first explain your reasoning towards solving the task and the tools that you want to use. Then in the 'Code' attribute, you should write the code in simple Python.
Do not write code that require importing packages. Generate a JSON object with the following structure:
```json
{
"thought": "...",
"code": "..."
}
```

The result of the code execution will be provided to you as 'Observation' as input for the next step.
In the end you have to return a final answer using the `final_answer` tool. 
Here is an example using notional tools:

Task: "What is the result of the following operation: 5 + 3 + 1294.678?"

{"thought": "I will use python code to compute the result of the operation and then return the final answer using the `final_answer` tool", "code": "result = 5 + 3 + 1294.678\nfinal_answer(result)\n"}

Your Tools:
"""
# custom executor output a triplet, (output, logs, is_final_answer)
custom_executor = LocalPythonExecutor([])

class ModelResponse(BaseModel):
    thought: str
    code: str

class SmallestPrimeFactorTool(Tool):
    name = "smallest_prime_factor"
    description = """Returns the smallest prime factor of n."""
    inputs = {
        "n": {
            "type": "integer",
            "description": "the number to find the smallest prime factor of",
        }
    }
    output_type = "integer"

    def forward(self, n: int) -> int:
        if n <= 1:
            return None
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return i
        return n  # n is prime if no factors found # n is prime if no factors found

class DivideTool(Tool):
    name = "divide"
    description = """Returns the quotient of n divided by d."""
    inputs = {
        "n": {
            "type": "integer",
            "description": "the number to divide",
        },
        "d": {
            "type": "integer",
            "description": "the divisor",
        }
    }
    output_type = "integer"

    def forward(self, n: int, d: int) -> int:
        if d == 0:
            raise ValueError("Cannot divide by zero.")
        return n // d

class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        return answer

final_answer_tool = FinalAnswerTool()
smallest_prime_factor_tool = SmallestPrimeFactorTool()
divide_tool = DivideTool()

class CodeActAgent (Agent):
    def __init__(self, tools=None, system_message=None, client="github", response_format=None):
        if system_message is None:
            system_message = prompt
            
        tool_descriptions = []
        for tool in tools:
            tool_descriptions.append(f"Tool: {tool.name}\nDescription: {tool.description}\nParameters: {json.dumps(tool.inputs, indent=2)}\nOutput: {tool.output_type}")
        
        system_message = system_message + "\n\nAvailable Tools:\n" + "\n\n".join(tool_descriptions)
        # Send tools to isolated python environment
        # https://github.com/huggingface/smolagents/blob/d12022bd05fb7d863fb8eb02f81f57e288ab0f36/src/smolagents/agents.py#L327C22-L327C57
        custom_executor.send_tools({tool.name: tool for tool in tools})
        
        super().__init__(system_message=system_message, client=client, response_format=response_format)

    def _action(self, snippet: str) -> bool:
        try:
            output, logs, is_final_answer = custom_executor(snippet)
            if is_final_answer:
                rprint(Text('Final Answer: ', style='bold magenta'), output)
                return True
            else:
                rprint(Text('Observation: ', style='bold green'), output)
                rprint(Text('Logs: ', style='bold blue'), logs)
                self.messages.append({"role": "user", "content": f"Observation: {output}\nLogs: {logs}"})
                return False
        except Exception as e:
            rprint(Text('Error: ', style='bold red'), e)


    def run(self, max=10):
        user_input = input("Question: ")
        self.messages.append({"role": "user", "content": "Question: " + user_input})
        step = 1
        while True or step < max:
            try:
                input("Press Enter to continue...")

                print_step_line(step)
                response = self.call(self.messages)
                step += 1
                if response.choices[0].message.content:
                    assistant_message = response.choices[0].message.content
                    formatted_message = {
                        "role": "assistant",
                        "content": assistant_message,
                    }
                    self.messages.append(formatted_message)
                    response = ModelResponse.model_validate_json(assistant_message)
                    rprint(Text('Thought: ', style='bold green'), response.thought)
                    rprint(Syntax(response.code, "python"))
                    is_final_answer = self._action(response.code)
                    if is_final_answer:
                        break

            except Exception as e:
                rprint(Text('Error: ', style='bold red'), e)

if __name__ == "__main__":
    tool_list = [smallest_prime_factor_tool, divide_tool, final_answer_tool]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "ThoughtAndCodeAnswer",
            "schema": {
                "additionalProperties": False,
                "properties": {
                    "thought": {
                        "description": "A free form text description of the thought process.",
                        "title": "Thought",
                        "type": "string",
                    },
                    "code": {
                        "description": "Valid Python code snippet implementing the thought.",
                        "title": "Code",
                        "type": "string",
                    },
                },
                "required": ["thought", "code"],
                "title": "ThoughtAndCodeAnswer",
                "type": "object",
            },
        },
    }


    agent = CodeActAgent(tools=tool_list, response_format=response_format, client="qwen")
    # print(custom_executor("x = 123"))
    # print(custom_executor("final_answer(x)"))

    agent.run()
    # print(final_answer(123))

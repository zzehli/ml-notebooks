import inspect
import json

from agent import Agent

# V1: this version is archived

# the purpose of the react agent is to make the agent carry out more difficult tasks through thought, action and observation loops.
# there are two ways of perform this: make llm generate a long response, with function calls mixed in the response; 2. make llm generate partial answer, stop at the function call, then feed the function call response to the llm to continue the loop
# Yao's implementation https://github.com/ysymyth/ReAct/blob/master/FEVER.ipynb
# note that Yao's paper does not use tool calling, the tool calls are parsed from the response, this is more flexible but less reliable.
# https://huggingface.co/datasets/alabnii/morehopqa
# need two llm calls per cycle, since function calls is one call by itself.
# notice that observations is part of the prompt, but in reality they are responses from the function calls.
# hallucination is baked in here

# three tools are provided: 1. smallest prime factor; 2. divide; 3. save factors

prompt = """
You are a helpful agent that can think, act and observe.
You will be given a question, and you will think step by step to solve the problem.
You will generate a thought, then an action, and then observe the result of the action.
Always start with a thought before call any tools.
You will repeat this process until you have a complete answer.
You will use the following tools:
1. smallest_prime_factor(n): returns the smallest prime factor of n
2. divide(n, d): returns the quotient of n divided by d
3. save_factors(num): saves the factor you found so far
4. get_factors(): returns the factors you have saved so far
You will use the tools to solve the problem step by step.

<example>
Question: What are the prime factors of 4?
Thought: I need to find the smallest prime factor of 4.
Action: [call smallest_prime_factor(4)]
Observation: The smallest prime factor of 4 is 2.
Thought: I've found a prime factor of 4, now I will save the result of the tool call, which is 2.
Action: [call save_factors(2)]
Observation: None
Thought: I've found the smallest prime factor of 4, next I will perform the same process for the quotient, which is 4 divided by 2.
Action: [call divide(4, 2)]
Observation: 2 is the quotient of 4 divided by 2.
Thought: I need to find the smallest prime factor of 2.
Action: [call smallest_prime_factor(2)]
Observation: The smallest prime factor of 2 is itself.
Thought: I've found a prime factor of 2, now I will save the result of the tool call, which is 2.
Action: [call save_factors(2)]
Observation: None
Thought: Because the smallest prime factor of 2 is itself, now I will return the factors I have found so far.
Action: [call get_factors()]
Observation: The factors I have found are: 2, 2.
Thought: The prime factors of 4 are 2, 2. Done.
</example>

<example>
Question: What are the prime factors of 5?
Thought: I need to find the smallest prime factor of 5.
Action: [call smallest_prime_factor(5)]
Observation: The smallest prime factor of 5 is itself.
Thought: I've found a prime factor of 5, now I will save the result of the tool call, which is 5.
Action: [call save_factors(5)]
Observation: None
Thought: Because the smallest prime factor of 5 is itself, now I will return the factors I have found so far.
Action: [call get_factors()]
Observation: The factors I have found so far are: 5.
Thought: The prime factors of 5 itself. Done.
</example>
"""


def smallest_prime_factor(n: int) -> int:
    """Returns the smallest prime factor of n."""
    if n <= 1:
        return None
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return i
    return n  # n is prime if no factors found


def divide(n: int, d: int) -> int:
    """Returns the quotient of n divided by d."""
    if d == 0:
        raise ValueError("Cannot divide by zero.")
    return n // d


_factors = []  # Module-level variable


def save_factors(num: int) -> None:
    """Saves the factor found so far."""
    global _factors
    _factors.append(num)


def get_factors() -> list[int]:
    """Returns the factors found so far."""
    return _factors


class ReactAgent(Agent):
    def __init__(self, tools=None, system_message=None, client="github"):
        super().__init__(tools=tools, system_message=system_message, client=client)
        self._action_sequence = ""

    def _thought(self):
        try:
            response = self.call(self.messages, stop=["Action:"])
            if response.choices[0].message.content:
                self._action_sequence += f"{response.choices[0].message.content}\n"
                return response.choices[0].message.content
            elif response.choices[0].message.tool_calls:
                return "Tool call"
            else:
                return "No response"
        except Exception as e:
            print(f"Error: {e}")
            raise e

    def _action(self, tool_call):
        tool_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        self._action_sequence += f"Action: {tool_name} with args {args}\n"
        tool_dict = {tool.__name__: tool for tool in self.tools}
        # the tool name has to be in the tool list
        func = tool_dict[tool_name]
        sig = inspect.signature(func)

        # Convert arguments to their expected types
        converted_args = {}
        for param_name, param in sig.parameters.items():
            if param_name in args:
                expected_type = (
                    param.annotation if param.annotation != inspect._empty else str
                )
                try:
                    converted_args[param_name] = expected_type(args[param_name])
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Error converting argument {param_name} to type {expected_type.__name__}: {e}"
                    )

        print(f"performing tool call: {tool_name} with args {converted_args}")
        try:
            res = func(**converted_args)
            print(f"result of the tool call: {res}")
            observation = f"Observation: the result of the {tool_name} with {converted_args} call is {res}\n"
            self._action_sequence += observation
            return res
        except Exception as e:
            raise ValueError(f"Error executing tool {tool_name} with args {args}: {e}")

    def run(self):
        user_input = input("Question: ")
        self.messages.append({"role": "user", "content": "Question: " + user_input})
        while True:
            try:
                input("Press Enter to continue...")

                thought_response = self._thought()
                if thought_response == "Done.":
                    break
                elif (
                    thought_response == "Tool call" or thought_response == "No response"
                ):
                    print(f"unexpected response: {thought_response}")
                    continue
                # print(f"action_sequence after thought: {self._action_sequence}")
                response = self.call(
                    [{"role": "assistant", "content": self._action_sequence}],
                    stop=["Observation:"],
                )
                if response.choices[0].message.tool_calls:
                    for tool_call in response.choices[0].message.tool_calls:
                        self._action(tool_call)
                print(
                    f"action_sequence after action and observation: {self._action_sequence}"
                )
                print("--------------------------------")
                self.messages.append(
                    {"role": "assistant", "content": self._action_sequence}
                )
                self._action_sequence = ""
            except Exception as e:
                print(f"Error: {e}")
                return "An error occurred while processing your request."
        # print the all messages in the messages list
        for message in self.messages:
            print(message)


if __name__ == "__main__":
    agent = ReactAgent(
        tools=[
            smallest_prime_factor,
            divide,
            save_factors,
            get_factors,
        ],
        system_message=prompt,
        client="google",
    )
    agent.run()

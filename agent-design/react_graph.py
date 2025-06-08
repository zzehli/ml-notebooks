import inspect
import json
from typing import Any, Dict, List

from graph import Graph, Node, State
from react_agent import divide, get_factors, prompt, save_factors, smallest_prime_factor
from utils import chat_with_tools

tools = [divide, save_factors, get_factors, smallest_prime_factor]
def create_react_agent():
    def agent_node(state: List[State]) -> Dict[str, Any]:
        """Agent node is the main node that will be used to generate the response."""
        
        if not state:
            chat_history = []
        else:
            chat_history = state[-1].state_content["chat_history"]

        print(f"chat_history: {chat_history}")
        response = chat_with_tools([{"role": "system", "content": prompt}, *chat_history],
                                    tools,
                                    stop=["Action"])
        print(f"response: {response}")
        tool_calls = None
        if response.content:
            chat_history.append({"role": "assistant", "content": response.content})
        elif response.tool_calls:
            tool_calls = response.tool_calls
        return {"chat_history": chat_history, "tool_calls": tool_calls}

    def tool_node(state: List[State]) -> Dict[str, Any]:
        """Tool node is responsible for executing the tool."""
        chat_history = state[-1].state_content["chat_history"]
        tool_calls = state[-1].state_content["tool_calls"]
        if chat_history[-1]["role"] != "assistant":
            raise ValueError("Tool node must receive a message from the assistant.")
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            tool_dict = {tool.__name__: tool for tool in tools}
            # the tool name has to be in the tool list
            func = tool_dict[tool_name]
            sig = inspect.signature(func)
            
            # Convert arguments to their expected types
            converted_args = {}
            for param_name, param in sig.parameters.items():
                if param_name in args:
                    expected_type = param.annotation if param.annotation != inspect._empty else str
                    try:
                        converted_args[param_name] = expected_type(args[param_name])
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Error converting argument {param_name} to type {expected_type.__name__}: {e}")
            
            print(f"performing tool call: {tool_name} with args {converted_args}")
            try:
                res = func(**converted_args)
                chat_history.append({"role": "user", "content": f"the result of the {tool_name} with {converted_args} call is {res}\n"})
            except Exception as e:
                raise ValueError(f"Error executing tool {tool_name} with args {args}: {e}")

        return chat_history
    
    def have_tool_call(state_content) -> str:
        """Have tool call node is responsible for checking if the assistant has a tool call."""
        chat_history = state_content["chat_history"]
        tool_calls = state_content["tool_calls"]
        if chat_history[-1]["role"] != "assistant":
            raise ValueError("Have tool call node must receive a message from the assistant.")
        if tool_calls:
            return "tool_call"
        elif "Done" in chat_history[-1]["content"]:
            return "end"
        else:
            return "agent"
    
    graph = Graph()
    agent = Node("agent_node", agent_node)
    tool = Node("tool_node", tool_node)
    end = Node("end", lambda x: None)
    graph.add_node(agent)
    graph.add_node(tool)
    graph.add_node(end)
    graph.add_conditional_edge(
        "agent_node",
        have_tool_call,
        {"tool_call": "tool_node", "end": "end", "agent": "agent_node"},
    )
    graph.add_edge("tool_node", "agent_node")
    agent.receive_message(State("initial", {"chat_history": [{"role": "user", "content": "What is the smallest prime factor of 100?"}]}))
    return graph

if __name__ == "__main__":
    # Create and run example graph
    graph = create_react_agent()
    
    # Run until no more messages are being processed
    res = graph.run()
    print("==================run trace====================")
    for m in res:
        print(m)

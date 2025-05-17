import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from utils import function_to_schema

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GITHUB_OPENAI_API_KEY"),
    base_url="https://models.github.ai/inference",
)

# Customer Service Routine

system_message = (
    "You are a customer support agent for ACME Inc."
    "Always answer in a sentence or less."
    "Follow the following routine with the user:"
    "1. First, ask probing questions and understand the user's problem deeper.\n"
    " - unless the user has already provided a reason.\n"
    "2. Propose a fix (make one up).\n"
    "3. ONLY if not satisfied, offer a refund.\n"
    "4. If accepted, search for the ID and then execute refund."
    ""
)


def look_up_item(search_query):
    """Use to find item ID.
    Search query can be a description or keywords."""

    # return hard-coded item ID - in reality would be a lookup
    return "item_132612938"


def execute_refund(item_id, reason="not provided"):
    print("Summary:", item_id, reason)  # lazy summary
    return "success"


def run_full_turn(system_message, messages):
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "system", "content": system_message}] + messages,
    )
    message = response.choices[0].message
    messages.append(message)

    if message.content:
        print("Assistant:", message.content)

    return message


tools = [execute_refund, look_up_item]


def run_full_turn_with_tools(system_message, tools, messages):
    num_init_messages = len(messages)
    messages = messages.copy()

    while True:
        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in tools]
        tools_map = {tool.__name__: tool for tool in tools}

        # === 1. get openai completion ===
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "system", "content": system_message}] + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)

        if message.content:  # print assistant response
            print("Assistant:", message.content)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # === 2. handle tool calls ===

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools_map)

            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            print("Call result: ", result_message)
            messages.append(result_message)

    # ==== 3. return new messages =====
    return messages[num_init_messages:]


def execute_tool_call(tool_call, tools_map):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"Assistant: {name}({args})")

    # call corresponding function with provided arguments
    return tools_map[name](**args)


def main():
    """Run the customer service agent with or without tools."""
    messages = []
    choice = input(
        "This is a customer service agent. Please choose 1 or 2: 1 for an agent with no tools, 2 for an agent with tools: "
    )
    if choice == "1":
        while True:
            user = input("User: ")
            messages.append({"role": "user", "content": user})

            run_full_turn(system_message, messages)
    elif choice == "2":
        while True:
            user = input("User: ")
            messages.append({"role": "user", "content": user})

            new_messages = run_full_turn_with_tools(system_message, tools, messages)
            messages.extend(new_messages)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()

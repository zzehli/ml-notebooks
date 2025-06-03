import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
)


class Agent:
    def __init__(self, tools=None):
        self.system_message = "you are a helpful assistant"
        self.messages = [{"role": "system", "content": self.system_message}]
        self.tools = tools if tools is not None else []
        self.client = InferenceClient(
            api_key=os.getenv("HUGGINGFACE_API_KEY"),
        )

    def run(self):
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Agent terminated.")
                    break
                self.messages.append({"role": "user", "content": user_input})
                response = self.client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct",
                    messages=self.messages,
                    tools=self.tools,
                )
                assistant_message = response.choices[0].message.content
                formatted_message = {"role": "assistant", "content": assistant_message}
                print(f"Assistant: {assistant_message}")
                self.messages.append(formatted_message)

            except Exception as e:
                print(f"Error: {e}")
                return "An error occurred while processing your request."


if __name__ == "__main__":
    agent = Agent()
    agent.run()

import inspect
import json
import re
from typing import List

from agent import Agent
from playwright.sync_api import TimeoutError, sync_playwright
from rich import print
from utils import print_step_line

prompt = """
You are a helpful agent that can think and use tools. Use the tools to solve the problem step by step.
When you use tools, always provide a message to explain your plan along with the tool call. When you trying to find information, only use the snapshot tool after you tried find_in_page tool. /no_think
"""


def get_current_page():
    """Get the current page."""
    if not browser.contexts:
        context = browser.new_context()
        return context.new_page()
    context = browser.contexts[0]  # Assuming you're using the default browser context
    if not context.pages:
        return context.new_page()
    return context.pages[-1]


browser = sync_playwright().start().chromium.launch(headless=False)
page = get_current_page()
elements = []


def navigate_to(url: str):
    """Go to a url."""
    page.goto(url)
    page.wait_for_load_state()
    return "Navigated to " + url


def snapshot() -> str:
    """Generate a snapshot of the current page, including all interactive elements, text, and links. This is slow, so use it sparingly."""
    global elements
    # Select only interactive elements directly
    
    elements = (
        page.locator("input, a, button, select, textarea, summary, h1, h2, h3, h4, h5, h6, p, span")
        .filter(visible=True)
        .all()
    )
    # match by div is too slow
    # elements = page.locator("div").filter(has=page.locator("h1, h2, h3, h4, h5, h6, p, span"), has_not=page.locator("header, footer, nav"), visible=True).all()
    if elements:
        res = {}
        for idx, elem in enumerate(elements[:100]):
            tag_name = elem.evaluate("el => el.tagName.toLowerCase()")

            if tag_name == "input":
                input_info = elem.evaluate("""el => ({
                    type: el.type,
                    value: el.value,
                    placeholder: el.placeholder,
                    role: el.role,
                    name: el.name
                })""")
                res[idx] = f"Input: {input_info}"
            elif tag_name == "a":
                link_info = elem.evaluate("""el => ({
                    href: el.href,
                    text: el.textContent.trim()
                })""")
                res[idx] = f"Link: {link_info}"
            else:
                text = tag_name + ": "
                text += elem.inner_text().strip()
                if text:
                    res[idx] = text
                else:
                    # If no text, get some basic attributes
                    attrs = elem.evaluate("""el => ({
                        id: el.id,
                        type: el.type,
                        name: el.name,
                        title: el.title,
                        placeholder: el.placeholder,
                        value: el.value,
                        text: el.textContent.trim()
                    })""")
                    res[idx] = f"Element: {attrs}"
        return f"Interactive elements found: {res}"
    else:
        return "No interactive elements found"


def find_links_with_text(text: str) -> str:
    """Find links with text."""
    global elements
    links = page.locator(f"a:has-text('{text}')").all()

    if links:
        elements = links
        res = {}
        for idx, elem in enumerate(links):
            res[idx] = elem.inner_text().strip()
        return f"Links found with text: {text}: {res}"
    else:
        return f"No links found with text: {text}"


# browser use click event: https://github.com/browser-use/browser-use/blob/79ca05f5340c667d077d296759a84be926127dc1/browser_use/browser/session.py#L1423-L1467
def click_element(index: int = 0) -> str:
    """Click an element, use the index number to indicate which element to interact with."""
    global elements
    old_url = page.url
    if elements:
        try:
            with page.expect_navigation(timeout=3000):
                elements[index].click()
                if page.url != old_url:
                    page.wait_for_load_state()
                    return "Element clicked, navigating to new page"
                return "Element clicked, no navigation occurred"
        except TimeoutError:
            if page.url == old_url:
                return "Element clicked, no navigation occurred"

    else:
        return "No elements was clicked"


def type_text(text: str, index: int = 0) -> str:
    """Type text into an element, use the index number to indicate which element to type."""
    global elements
    if elements:
        elements[index].fill(text)
        page.wait_for_timeout(1500)
        return f"Type '{text}' into element {index}."
    else:
        return "No element was typed into"


def find_in_page(keyword: str) -> List[str]:
    """Look for information by searching a keyword and return sentences that contain the keywords."""
    # Get all elements containing the keyword
    elements = (
        page.get_by_text(re.compile(keyword, re.IGNORECASE), exact=False)
        .filter(visible=True)
        .all()
    )

    # Extract text from each element and filter out empty strings
    sentences = []
    for element in elements:
        # element.highlight()
        text = element.inner_text().strip()
        if text:
            sentences.append(text)

    return f"The following sentences are found: {sentences}"

def user_input(feedback: str):
    """Solicit user input when the agent needs to know more about the current task,unclear how to proceed or is finished with the task."""
    user_input = input("Feedback: ")
    return f"User input: {user_input}"

def close():
    """Close the browser."""
    browser.close()


class ReactAgent(Agent):
    def __init__(self, tools=None, system_message=None, client="github"):
        super().__init__(tools=tools, system_message=system_message, client=client)

    def _action(self, tool_call):
        tool_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
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
            observation = (
                f"The result of the {tool_name} with {converted_args} call is {res}"
            )
            print(observation)
            self.messages.append({"role": "user", "content": observation})
            return res
        except Exception as e:
            raise ValueError(f"Error executing tool {tool_name} with args {args}: {e}")

    def run(self, max=10):
        user_input = input("Question: ")
        self.messages.append({"role": "user", "content": "Question: " + user_input})
        step = 1
        while True or step < max:
            try:
                input("Press Enter to continue...")

                print_step_line(step)
                response = self.call(self.messages)

                if response.choices[0].message.content:
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    )
                    print("assistant: " + response.choices[0].message.content)
                if response.choices[0].message.tool_calls:
                    for tool_call in response.choices[0].message.tool_calls:
                        self._action(tool_call)
                step += 1
            except Exception as e:
                print(f"Error: {e}")
                return "An error occurred while processing your request."
        # print the all messages in the messages list
        # for message in self.messages:
        #     print(message)


if __name__ == "__main__":
    agent = ReactAgent(
        tools=[
            navigate_to,
            snapshot,
            find_in_page,
            type_text,
            click_element,
            user_input,
        ],
        system_message=prompt,
        client="github",
    )
    agent.run()
    # print(navigate_to("https://www.nytimes.com/"))
    # print(snapshot())
    close()

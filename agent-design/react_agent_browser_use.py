# https://github.com/ysymyth/ReAct/blob/master/wikienv.py
# https://colab.research.google.com/drive/1OqYAKT1OcAiQgIRE5PAAHBI4CB2lG-4n?usp=sharing#scrollTo=mAFWR3TeHkJs

import re
import time
from typing import List

from playwright.sync_api import TimeoutError, sync_playwright
from react_agent import ReactAgent

prompt = """You are a helpful assistant that can use the Playwright library to interact with a browser. Use the browser to find information for user questions. 
You will be given a question, and you will think step by step to solve the problem.
You will generate a thought, then an action, and then observe the result of the action.
You have a list of tools to navigate the web. Prioritize finding the information through wikipedia.org.

To use these tools effectively:
1. First navigate to a page using navigate_to()
2. Use find_elements() or find_links_with_text() to locate elements
3. Interact with elements using click_element() or type_text()
4. Use find_in_page() to search for specific content

Remember:
- Elements are stored globally after finding them
- Click operations may trigger navigation
- Some operations may require waiting for page loads
- Input elements need type_text() rather than click_element()


<example>
Question: When was Python Playwright launched?
Thought: I should search for Python's version on python.org
Action: [call navigate_to("https://www.wikipedia.org/")]
Observation: the result of the navigate_to with {'url': 'https://www.wikipedia.org'} call is None

Thought: I need to find the search bar
Action: [call find_elements("input[name='search'], #searchInput")]
Observation: Elements found with selector: input[name='search'], #searchInput: {0: "Input: {'type': 'search', 'value': '', 'placeholder': 'Search Wikipedia', 'name': 'search'}", 1: "Input: {'type': 'search', 'value': '', 'placeholder': 'Search Wikipedia', 'name': 'search'}"}

Thought: The index 0 is the search bar. I'll type in "Playwright"
Action: [call type_text("Playwright", 0)]
Observation: Type 'Playwright' into element 0.

Thought: There will be many results about Playwright. I should find the links with text "Playwright"
Action: [call find_links_with_text("Playwright")]
Observation: Links found with text: Playwright: Links found with text: Playwright: {0: 'Playwright\nPerson who writes plays', 1: 'Playwright (software)\nEnd-to-end testing framework', 2: 'Playwrights Horizons\nOff-Broadway theater in Manhattan, New York', 3: 'Playwrights Guild of Canada', 4: "Playwrights' Company\nFormer American theatrical production company", 5: "Playwrights '56\nAmerican dramatic anthology TV series", 6: 'Playwrights Canada Press\nCanadian publishing house', 7: 'Playwrights Horizons Theater School', 8: "Playwrights' Advisory Board\nAustralian organisation", 9: 'Playwrights Theatre Club\nInfluential 1950s Chicago theater', 10: 'Search for pages containing Playwright'}

Thought: I should go to the wikipedia page for Playwright Python package, which has index 1.
Action: [call click_element(1)]
Observation: Element clicked, navigating to new page

Thought: I need to find the initial release section to see the launch date
Action: [call find_in_page("launch")]
Observation: The following sentences are found: ['Playwright is an open-source automation library for browser testing and web scraping[3] developed by Microsoft[4][5] and launched on 31 January 2020, which has since become popular among programmers and web developers.']

Thought: I've found the launch date. The launch date is 31 January 2020. Done.
</example>
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

def find_elements(selector: str) -> str:
    """Find elements by selector."""
    global elements
    elements = page.locator(selector).all()
    if elements:
        res = {}
        for idx, elem in enumerate(elements):
            tag_name = elem.evaluate('el => el.tagName.toLowerCase()')
            
            if tag_name == 'input':
                input_info = elem.evaluate('''el => ({
                    type: el.type,
                    value: el.value,
                    placeholder: el.placeholder,
                    name: el.name
                })''')
                res[idx] = f"Input: {input_info}"
            elif tag_name == 'a':
                link_info = elem.evaluate('''el => ({
                    href: el.href,
                    text: el.textContent.trim()
                })''')
                res[idx] = f"Link: {link_info}"
            else:
                text = elem.inner_text().strip()
                if text:
                    res[idx] = text
                else:
                    # If no text, get some basic attributes
                    attrs = elem.evaluate('''el => ({
                        id: el.id,
                        type: el.type
                    })''')
                    res[idx] = f"Element: {attrs}"
        return f"Elements found with selector: {selector}: {res}"
    else:
        return f"No elements found with selector: {selector}"
    
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
    """Click an element, if no index is provided, click the first element."""
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
        except TimeoutError as e:
            if page.url == old_url:
                return "Element clicked, no navigation occurred"

    else:
        return "No elements was clicked"

def type_text(text: str, index: int = 0) -> str:
    """Type text into an element, if no index is provided, type text into the first element."""
    global elements
    if elements:
        elements[index].fill(text)
        page.wait_for_timeout(1500)
        return f"Type '{text}' into element {index}."
    else:
        return "No element was typed into"

def find_in_page(keyword: str) -> List[str]:
    """Search a keyword and return sentences that contain the keywords."""
    # Get all elements containing the keyword
    elements = page.get_by_text(re.compile(keyword, re.IGNORECASE), exact=False).all()
    
    # Extract text from each element and filter out empty strings
    sentences = []
    for element in elements:
        # element.highlight()
        text = element.inner_text().strip()
        if text:
            sentences.append(text)
    
    return f"The following sentences are found: {sentences}"

def close():
    """Close the browser."""
    browser.close()
if __name__ == "__main__":
    agent = ReactAgent(
        tools=[
            find_in_page,
            click_element,
            type_text,
            find_elements,
            navigate_to,
            find_links_with_text,
        ],
        system_message=prompt,
        client="google"
    )

    agent.run()
    close()

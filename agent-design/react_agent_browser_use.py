# https://github.com/ysymyth/ReAct/blob/master/wikienv.py
# https://colab.research.google.com/drive/1OqYAKT1OcAiQgIRE5PAAHBI4CB2lG-4n?usp=sharing#scrollTo=mAFWR3TeHkJs

# go to website

# lookup keyword -> return sentences that contain the keywords

# search links that contain keywords

# click links
import time
from typing import List

from playwright.sync_api import TimeoutError, sync_playwright
from react_agent import ReactAgent

# with sync_playwright() as p:
#     browser = p.chromium.launch(headless=False)  # Set headless=False to see the browser window
#     page = browser.new_page()
#     page.goto("https://example.com")
    
#     # Keep the browser open for 10 seconds so you can see it
#     time.sleep(10)
    
    # browser.close()

# T = TypeVar("T")

# def run_async(coro: Coroutine[Any, Any, T]) -> T:
#     loop = asyncio.get_event_loop()
#     return loop.run_until_complete(coro)

prompt = """
You are a helpful assistant that can use the Playwright library to interact with a browser.
"""
# browser = sync_playwright().start().chromium.launch(headless=False)
# page
class PlaywrightTool:
    def __init__(self):
        self.browser = sync_playwright().start().chromium.launch(headless=False)
        self.page = self._get_current_page()
        self.elements = []

    def _get_current_page(self):
        """Get the current page."""
        if not self.browser.contexts:
            context = self.browser.new_context()
            return context.new_page()
        context = self.browser.contexts[0]  # Assuming you're using the default browser context
        if not context.pages:
            return context.new_page()
        return context.pages[-1]

    def navigate_to(self, url: str):
        """Go to a url."""
        self.page.goto(url)
        self.page.wait_for_load_state()

    def find_elements(self, selector: str) -> str:
        """Find elements by selector."""
        # TODO: return a list of elements so the model can choose
        self.elements = self.page.locator(selector).all()
        print(self.elements)
        if self.elements:
            res = {}
            for idx, elem in enumerate(self.elements):
                res[idx] = elem.text_content().strip()
            return f"Elements found with selector: {selector}: {res}"
        else:
            return f"No elements found with selector: {selector}"
    
    def find_links_with_text(self, text: str) -> str:
        """Find links with text."""
        links = self.page.locator(f"a:has-text('{text}')").all()

        if links:
            self.elements = links
            res = {}
            for idx, elem in enumerate(links):
                res[idx] = elem.inner_text().strip()
            return f"Links found with text: {text}: {res}"
        else:
            return f"No links found with text: {text}"
    # browser use click event: https://github.com/browser-use/browser-use/blob/79ca05f5340c667d077d296759a84be926127dc1/browser_use/browser/session.py#L1423-L1467
    def click_element(self, index: int = 0) -> str:
        """Click an element, if no index is provided, click the first element."""
        old_url = self.page.url
        if self.elements:
            try:
                with self.page.expect_navigation(timeout=3000):
                    self.elements[index].click()
                    if self.page.url != old_url:
                        self.page.wait_for_load_state()
                        return "Element clicked, navigating to new page"
                    return "Element clicked, no navigation occurred"
            except TimeoutError as e:
                if self.page.url == old_url:
                    return "Element clicked, no navigation occurred"

        else:
            return "No elements was clicked"

    def type_text(self, text: str, index: int = 0) -> str:
        """Type text into an element, if no index is provided, type text into the first element."""
        if self.elements:
            self.elements[index].fill(text)
            return f"Type '{text}' into element: {self.elements[index]}"
        else:
            return "No element was typed into"

    def find_in_page(self, keyword: str) -> List[str]:
        """Search a keyword and return sentences that contain the keywords."""
        # Get all elements containing the keyword
        # elements = self.page.locator(f"//*[contains(text(), '{keyword}')]").all()
        elements = self.page.locator(
            f"//*[not(self::script) and not(self::style) and not(self::noscript) and contains(text(), '{keyword}')]"
        ).all()
        # Extract text from each element and filter out empty strings
        sentences = []
        for element in elements:
            # element.highlight()
            text = element.inner_text().strip()
            if text:
                sentences.append(text)
        
        # Store elements for potential future use (like clicking)
        self.elements = elements
        
        return f"The following sentences are found: {sentences}"

    def close(self):
        """Close the browser."""
        self.browser.close()
if __name__ == "__main__":
    agent = ReactAgent(
        tools=[
            PlaywrightTool.find_in_page,
            PlaywrightTool.click_element,
            PlaywrightTool.type_text,
            PlaywrightTool.find_elements,
            PlaywrightTool.navigate_to,
        ],
        system_message=prompt,
        client="google"
    )
    playwright_tool = PlaywrightTool()
    playwright_tool.navigate_to("https://en.wikipedia.org/wiki/Playwright_(software)")
    print(playwright_tool.find_elements("input[name='search'], #searchInput"))
    print(playwright_tool.click_element())
    print(playwright_tool.type_text("Playwright"))
    time.sleep(5)
    print(playwright_tool.find_elements("a:has-text('Playwright')"))
    print(playwright_tool.click_element())
    time.sleep(5)
    print(playwright_tool.find_in_page("Ancient Greeks"))
    print(playwright_tool.find_elements("input[name='search'], #searchInput"))


    playwright_tool.close()
    # agent.run()
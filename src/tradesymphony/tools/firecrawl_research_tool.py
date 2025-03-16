import requests
from bs4 import BeautifulSoup
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import logging
from typing import Type

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FirecrawlResearchInput(BaseModel):
    url: str = Field(
        description="The URL of the website to crawl and extract information from."
    )


class FirecrawlResearchTool(BaseTool):
    name: str = "FirecrawlResearchTool"
    description: str = (
        "Useful for crawling specific websites and extracting relevant information. "
        "Input should be the URL of the website to crawl."
    )
    args_schema: Type[BaseModel] = FirecrawlResearchInput

    def _run(self, url: str) -> str:
        """Use the tool."""
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract all text from the webpage
            text = soup.get_text(separator=" ", strip=True)

            return f"Crawling website: {url}. Content: {text[:500]}..."  # Limit output for brevity

        except requests.exceptions.RequestException as e:
            return f"Could not crawl website: {url}. Error: {str(e)}"

    async def _arun(self, url: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution")

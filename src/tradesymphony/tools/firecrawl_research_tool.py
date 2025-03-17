import os
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List, Optional
import asyncio
from firecrawl import FirecrawlApp


class FirecrawlResearchInput(BaseModel):
    """Input model for the FirecrawlResearchTool."""

    url: str = Field(description="The URL to crawl.")
    formats: List[str] = Field(
        default=["markdown"], description="List of formats to return."
    )
    actions: List[str] = Field(
        default=[], description="List of actions to perform on the page."
    )
    timeout: Optional[int] = Field(
        default=30000, description="Timeout in milliseconds for the scraping operation."
    )


class FirecrawlResearchTool(BaseTool):
    name: str = "FirecrawlResearchTool"
    description: str = (
        "Tool for crawling and extracting content from a website using Firecrawl. "
        "It accepts a URL and optional formats and actions to perform on the page."
    )
    args_schema: Type[BaseModel] = FirecrawlResearchInput

    def _run(
        self,
        url: str,
        formats: list = ["markdown"],
        actions: list = [],
        timeout: int = 30000,
    ) -> str:
        """Use the tool."""
        try:
            api_key = os.getenv("FIRECRAWL_API_KEY")
            if not api_key:
                return "Error: FIRECRAWL_API_KEY environment variable not set."

            app = FirecrawlApp(api_key=api_key)
            scrape_result = app.scrape_url(
                url, params={"formats": formats, "actions": actions, "timeout": timeout}
            )

            if scrape_result and scrape_result.get("success"):
                return str(
                    scrape_result["data"]
                )  # Ensure the data is converted to a string
            else:
                error_message = (
                    scrape_result.get("error", "No additional error details provided.")
                    if scrape_result
                    else "Unknown error occurred."
                )
                return f"Error: Could not retrieve content from {url}. Result: {scrape_result}. Error details: {error_message}"

        except Exception as e:
            return f"Error: Could not retrieve content from {url}. {str(e)}"

    async def _arun(self, *args, **kwargs):
        return await asyncio.to_thread(self._run, *args, **kwargs)

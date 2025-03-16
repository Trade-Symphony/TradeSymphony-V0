from tavily import TavilyClient
import os
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import logging
from typing import Type

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TavilySearchInput(BaseModel):
    query: str = Field(
        description="The search query to look up information on the web."
    )


class TavilySearchTool(BaseTool):
    name: str = "TavilySearchTool"
    description: str = (
        "Useful for searching the web and retrieving information from various sources. "
        "Input should be a specific search query."
    )
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str) -> str:
        """Use the tool."""
        try:
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                return "Tavily API key not found in environment variables."

            client = TavilyClient(api_key=tavily_api_key)
            search_results = client.search(query)
            return str(search_results)
        except Exception as e:
            return f"Could not retrieve search results for {query}. Error: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution")

from tavily import TavilyClient
import os
from crewai.tools import BaseTool
import logging
from pydantic import BaseModel, Field
from typing import Type

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompanyResearchInput(BaseModel):
    """Input schema for CompanyResearchTool.

    This model defines the required and optional parameters for company research requests.
    """

    company: str = Field(
        ..., description="The company name or ticker symbol to research"
    )


class CompanyResearchTool(BaseTool):
    name: str = "CompanyResearchTool"
    description: str = (
        "Useful for gathering information about a specific company, including financials, news, and SEC filings. "
        "Input should be the company name or ticker symbol."
    )
    args_schema: Type[BaseModel] = CompanyResearchInput

    def _run(self, company: str) -> str:
        """Use the tool."""
        try:
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                return "Tavily API key not found in environment variables."

            client = TavilyClient(api_key=tavily_api_key)
            search_results = client.search(f"{company} company information")

            return f"Gathering information about company: {company}. Search results: {search_results}"

        except Exception as e:
            return f"Could not gather information about company: {company}. Error: {str(e)}"

    async def _arun(self, company: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution")

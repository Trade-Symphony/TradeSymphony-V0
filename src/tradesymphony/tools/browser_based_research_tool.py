import os
import json
from typing import Dict, Any, Type, Optional
from crewai.tools import BaseTool
from datetime import datetime
import logging
from pydantic import BaseModel, Field, root_validator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrowserBasedResearchInput(BaseModel):
    url: Optional[str] = Field(
        default=None, description="URL to browse and analyze for financial information"
    )
    query: Optional[str] = Field(
        default=None,
        description="Search query for finding market and financial information",
    )
    depth: int = Field(
        default=1,
        description="Search depth (1 for basic search, > 1 for advanced search)",
        ge=1,
    )

    @root_validator(pre=True)
    def check_url_or_query(cls, values):
        if not values.get("url") and not values.get("query"):
            raise ValueError("Either 'url' or 'query' must be provided")
        return values


class BrowserBasedResearchTool(BaseTool):
    name: str = "BrowserBasedResearchTool"
    description: str = (
        "Market research tool for browsing websites, gathering data, and analyzing financial information. "
        "Uses Tavily for search capabilities and ColiVara for document analysis. "
        "Input format: {'url': 'https://example.com', 'query': 'financial data', 'depth': 2}"
    )
    args_schema: Type[BaseModel] = BrowserBasedResearchInput

    def __init__(self):
        """Initialize the BrowserBasedResearchTool with API keys."""
        super().__init__()
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.colivara_api_key = os.getenv("COLIVARA_API_KEY")
        self.colivara_client = None

        # Initialize ColiVara client if API key is available
        if self.colivara_api_key:
            try:
                from colivara_py import ColiVara

                self.colivara_client = ColiVara(api_key=self.colivara_api_key)
                logger.info("ColiVara client initialized successfully")
            except ImportError:
                logger.warning(
                    "ColiVara SDK not installed. Run 'pip install colivara-py' to enable this feature."
                )
            except Exception as e:
                logger.error(f"Failed to initialize ColiVara client: {str(e)}")

    def _run(self, url, query, depth) -> str:
        """Run the advanced research tool."""
        try:
            if not url and not query:
                return "Please provide either a URL to browse or a query to search."

            results = {}

            # Use Tavily for search if query is provided
            if query and self.tavily_api_key:
                try:
                    from tavily import TavilyClient

                    client = TavilyClient(api_key=self.tavily_api_key)
                    search_results = client.search(
                        query=query,
                        search_depth="advanced" if depth > 1 else "basic",
                        include_images=False,
                        include_answer=True,
                        max_results=5,
                    )
                    results["tavily_search"] = search_results
                except Exception as e:
                    logger.error(f"Tavily search error: {str(e)}")
                    results["tavily_search_error"] = str(e)

            # Use ColiVara for document analysis if API key is available and URL is provided
            if url and self.colivara_client:
                try:
                    # Use ColiVara's proper SDK instead of direct API calls
                    collection_name = "market_research"

                    # Use the URL directly - ColiVara can handle webpage screenshots
                    document = self.colivara_client.upsert_document(
                        name=f"research_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        url=url,
                        collection_name=collection_name,
                        metadata={"query": query} if query else {},
                    )

                    # Search the document with the query or a default query
                    search_query = (
                        query if query else "extract key financial information"
                    )
                    search_results = self.colivara_client.search(
                        query=search_query, collection_name=collection_name, top_k=3
                    )

                    # Format search results
                    colivara_results = []
                    for result in search_results.results:
                        colivara_results.append(
                            {
                                "page": result.page_number,
                                "document": result.document_name,
                                "score": result.normalized_score,
                                "content": result.content
                                if hasattr(result, "content")
                                else "Content not available",
                            }
                        )

                    results["colivara_analysis"] = {
                        "document_id": document.id,
                        "document_name": document.name,
                        "search_results": colivara_results,
                    }

                except Exception as e:
                    logger.error(f"ColiVara integration error: {str(e)}")
                    results["colivara_error"] = str(e)

            # Format and return the combined results
            return json.dumps(
                {
                    "query": query,
                    "url": url,
                    "depth": depth,
                    "timestamp": datetime.now().isoformat(),
                    "results": results,
                },
                indent=2,
            )

        except Exception as e:
            logger.error(f"BrowserBasedResearchTool error: {str(e)}")
            return f"Error in research tool: {str(e)}"

    async def _arun(self, input_data: Dict[str, Any] | str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution")

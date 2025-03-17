from crewai.tools import BaseTool
import yfinance as yf
import json
import pandas as pd
from pydantic import BaseModel, Field
from typing import Dict, Any, Type
import asyncio

try:
    from browserbase import BrowserBase
except ImportError:
    BrowserBase = None
    print(
        "BrowserBase not installed. To use BrowserBaseTool, install browserbase package"
    )


# class StockScreenerInput(BaseModel):
#     market_cap_min: Optional[float] = Field(
#         None, description="Minimum market capitalization in USD"
#     )
#     market_cap_max: Optional[float] = Field(
#         None, description="Maximum market capitalization in USD"
#     )
#     pe_ratio_min: Optional[float] = Field(
#         None, description="Minimum price-to-earnings ratio"
#     )
#     pe_ratio_max: Optional[float] = Field(
#         None, description="Maximum price-to-earnings ratio"
#     )
#     dividend_yield_min: Optional[float] = Field(
#         None, description="Minimum dividend yield (as decimal)"
#     )
#     dividend_yield_max: Optional[float] = Field(
#         None, description="Maximum dividend yield (as decimal)"
#     )
#     price_min: Optional[float] = Field(None, description="Minimum stock price in USD")
#     price_max: Optional[float] = Field(None, description="Maximum stock price in USD")
#     volume_min: Optional[int] = Field(None, description="Minimum trading volume")
#     sector: Optional[str] = Field(
#         None,
#         description="Specific sector to filter by (e.g. 'Technology', 'Healthcare')",
#     )
#     industry: Optional[str] = Field(None, description="Specific industry to filter by")
#     custom_criteria: Optional[Dict[str, Any]] = Field(
#         None,
#         description="Any additional custom screening criteria in format {metric_min/max: value}",
#     )

#     def to_dict(self) -> Dict[str, Any]:
#         """Convert the model to a dictionary of criteria to use in the stock screener."""
#         result = {}
#         for key, value in self.dict(
#             exclude_none=True, exclude={"custom_criteria"}
#         ).items():
#             if value is not None:
#                 result[key] = value

#         # Add any custom criteria
#         if self.custom_criteria:
#             result.update(self.custom_criteria)


#         return result
class StockScreenerInput(BaseModel):
    criteria: dict = Field(..., description="Screening criteria for stocks")
    # Add other parameters that the _run method expects


class StockScreenerTool(BaseTool):
    name: str = "StockScreenerTool"
    description: str = (
        "Useful for screening stocks based on various financial metrics. "
        "Provide criteria like market cap, P/E ratio, dividend yield, etc. "
        "Format: {'market_cap_min': 1000000000, 'pe_ratio_max': 20, 'dividend_yield_min': 0.02}"
    )

    args_schema: Type[BaseModel] = StockScreenerInput

    def _run(self, criteria: Dict[str, Any]) -> str:
        """Use the tool."""
        try:
            # Parse criteria from string to dict if provided as string
            if isinstance(criteria, str):
                try:
                    criteria = json.loads(criteria.replace("'", '"'))
                except json.JSONDecodeError:
                    return "Invalid criteria format. Please provide a JSON object."

            # Default S&P 500 tickers
            sp500_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
            sp500 = pd.read_csv(sp500_url)
            tickers = sp500["Symbol"].tolist()[:50]  # Limit to first 50 for performance

            results = []

            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info

                    # Check if stock meets criteria
                    meets_criteria = True
                    for key, value in criteria.items():
                        if key.endswith("_min"):
                            metric = key.replace("_min", "")
                            if metric in info and info[metric] is not None:
                                if float(info[metric]) < float(value):
                                    meets_criteria = False
                                    break
                        elif key.endswith("_max"):
                            metric = key.replace("_max", "")
                            if metric in info and info[metric] is not None:
                                if float(info[metric]) > float(value):
                                    meets_criteria = False
                                    break

                    if meets_criteria:
                        results.append(
                            {
                                "symbol": ticker,
                                "name": info.get("longName", "Unknown"),
                                "sector": info.get("sector", "Unknown"),
                                "industry": info.get("industry", "Unknown"),
                                "market_cap": info.get("marketCap", "N/A"),
                                "pe_ratio": info.get("trailingPE", "N/A"),
                                "dividend_yield": info.get("dividendYield", "N/A"),
                            }
                        )

                except Exception:
                    continue

            if not results:
                return "No stocks found matching your criteria."

            return json.dumps(results)
        except Exception as e:
            return f"Error performing stock screening: {str(e)}"

    async def _arun(self, *args, **kwargs):
        return await asyncio.to_thread(self._run, *args, **kwargs)

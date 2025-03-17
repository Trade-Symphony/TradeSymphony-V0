from crewai.tools import BaseTool
import yfinance as yf
import json
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Type
import asyncio


class FinancialDataInput(BaseModel):
    ticker: str = Field(
        ..., description="Stock ticker symbol to fetch financial data for"
    )


class FinancialDataTool(BaseTool):
    name: str = "FinancialDataTool"
    description: str = (
        "Useful for fetching comprehensive financial data for a given stock ticker. "
        "Provide the stock ticker symbol."
    )
    args_schema: Type[BaseModel] = FinancialDataInput

    def _run(self, ticker: str) -> str:
        """Use the tool."""
        try:
            stock = yf.Ticker(ticker)

            # First check if we can get basic history data
            hist = stock.history(period="1d")
            if hist.empty:
                return json.dumps(
                    {
                        "error": True,
                        "message": f"Invalid ticker symbol: {ticker}. This symbol may not exist or may have been delisted.",
                    },
                    indent=2,
                )

            info = stock.info

            # Check if info contains meaningful data (not just trailingPegRatio)
            if len(info) <= 1 or (len(info) == 1 and "trailingPegRatio" in info):
                return json.dumps(
                    {
                        "error": True,
                        "message": f"Could not retrieve meaningful data for {ticker}. This may not be a valid ticker symbol.",
                    },
                    indent=2,
                )

            # Get historical data
            hist = stock.history(period="6mo")
            recent_price = hist["Close"].iloc[-1] if not hist.empty else None
            # Get financial statements
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow

            # Key financial metrics
            financial_data = {
                "symbol": stock.ticker,
                "company_name": info.get("longName", "Unknown"),
                "summary": info.get("longBusinessSummary", "No description available"),
                "industry": info.get("industry", "Unknown"),
                "sector": info.get("sector", "Unknown"),
                "market_cap": info.get("marketCap", None),
                "pe_ratio": info.get("trailingPE", None),
                "forward_pe": info.get("forwardPE", None),
                "price_to_sales": info.get("priceToSalesTrailing12Months", None),
                "price_to_book": info.get("priceToBook", None),
                "dividend_yield": info.get("dividendYield", None) * 100
                if info.get("dividendYield")
                else None,
                "eps": info.get("trailingEps", None),
                "beta": info.get("beta", None),
                "52_week_high": info.get("fiftyTwoWeekHigh", None),
                "52_week_low": info.get("fiftyTwoWeekLow", None),
                "50_day_average": info.get("fiftyDayAverage", None),
                "200_day_average": info.get("twoHundredDayAverage", None),
                "current_price": recent_price,
                "cash_flow": cash_flow.to_string(),
            }

            # Add income statement highlights if available
            if not income_stmt.empty:
                try:
                    latest_year = income_stmt.columns[0]
                    financial_data["revenue"] = income_stmt.loc[
                        "Total Revenue", latest_year
                    ]
                    financial_data["operating_income"] = income_stmt.loc[
                        "Operating Income", latest_year
                    ]
                    financial_data["net_income"] = income_stmt.loc[
                        "Net Income", latest_year
                    ]
                except (KeyError, IndexError):
                    pass

            # Add balance sheet highlights if available
            if not balance_sheet.empty:
                try:
                    latest_quarter = balance_sheet.columns[0]
                    financial_data["total_assets"] = balance_sheet.loc[
                        "Total Assets", latest_quarter
                    ]
                    financial_data["total_debt"] = (
                        balance_sheet.loc["Total Debt", latest_quarter]
                        if "Total Debt" in balance_sheet.index
                        else None
                    )
                    financial_data["total_equity"] = balance_sheet.loc[
                        "Total Stockholder Equity", latest_quarter
                    ]
                except (KeyError, IndexError):
                    pass

            # Recent news
            news = stock.news
            if news:
                financial_data["recent_news"] = [
                    {
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "publisher": item.get("publisher"),
                        "published": datetime.fromtimestamp(
                            item.get("providerPublishTime")
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        if item.get("providerPublishTime")
                        else None,
                    }
                    for item in news[:3]  # Limit to 3 news items
                ]

            return json.dumps(financial_data, indent=2, default=str)
        except Exception as e:
            return f"Could not retrieve financial data for {ticker}. Error: {str(e)}"

    async def _arun(self, *args, **kwargs):
        return await asyncio.to_thread(self._run, *args, **kwargs)

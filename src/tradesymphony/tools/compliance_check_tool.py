from crewai.tools import BaseTool
import yfinance as yf
import json
from typing import List, Type
from datetime import datetime
from pydantic import BaseModel, Field
import asyncio


class ComplianceCheckInput(BaseModel):
    """Input schema for Compliance Check Tool."""

    action: str = Field(..., description="Trading action to perform ('buy' or 'sell')")

    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL', 'MSFT')")

    quantity: int = Field(..., description="Number of shares to buy or sell", gt=0)

    client_type: str = Field(
        ..., description="Type of client ('retail' or 'institutional')"
    )

    restrictions: List[str] = Field(
        default=[],
        description="List of investment restrictions to check (e.g., ['ESG', 'sector'])",
    )


class ComplianceCheckTool(BaseTool):
    name: str = "ComplianceCheckTool"
    description: str = (
        "Tool for checking if an investment decision complies with regulations and internal policies. "
        "Input format: {'action': 'buy/sell', 'ticker': 'AAPL', 'quantity': 100, 'client_type': 'retail/institutional', 'restrictions': ['ESG', 'sector']}"
    )
    args_schema: Type[BaseModel] = ComplianceCheckInput

    def _run(self, action, ticker, quantity, client_type, restrictions) -> str:
        """Use the tool."""
        try:
            if not all([action, ticker, quantity, client_type]):
                return "Missing required fields. Please provide action, ticker, quantity, and client_type."

            # Basic validation
            if action.lower() not in ["buy", "sell"]:
                return "Invalid action. Must be 'buy' or 'sell'."

            if quantity <= 0:
                return "Invalid quantity. Must be greater than 0."

            if client_type.lower() not in ["retail", "institutional"]:
                return "Invalid client type. Must be 'retail' or 'institutional'."

            # Get stock information for compliance checks
            try:
                stock_info = yf.Ticker(ticker)
                info = stock_info.info
                if not info or "regularMarketPrice" not in info:
                    return f"Could not retrieve information for ticker {ticker}."

                # Extract relevant information
                market_price = info.get("regularMarketPrice", 0)
                market_cap = info.get("marketCap", 0)
                average_volume = info.get("averageDailyVolume10Day", 0)
                sector = info.get("sector", "Unknown")
                industry = info.get("industry", "Unknown")
                esg_scores = info.get("esgScores", {})

                # Initialize compliance checks
                compliance_issues = []
                warnings = []

                # Check for market cap restrictions
                if client_type.lower() == "retail":
                    # For retail clients, warn if investing in micro-cap stocks
                    if market_cap < 300000000:  # $300 million threshold for micro-cap
                        warnings.append(
                            f"Warning: {ticker} is a micro-cap stock (market cap: ${market_cap:,}), which may have higher volatility and risk."
                        )

                    # Check order size relative to average volume (retail clients limited to 5%)
                    if quantity * market_price > 0.05 * average_volume * market_price:
                        compliance_issues.append(
                            "Order size exceeds 5% of average daily volume for retail clients."
                        )
                else:  # Institutional
                    # For institutional clients, different thresholds apply (10%)
                    if quantity * market_price > 0.10 * average_volume * market_price:
                        warnings.append(
                            f"Order size represents significant portion of daily volume ({(quantity * market_price / (average_volume * market_price) * 100):.2f}%), consider staged execution."
                        )

                # Check ESG restrictions
                if "ESG" in restrictions and esg_scores:
                    total_esg = esg_scores.get("totalEsg", 0)
                    if total_esg < 30:  # Assuming lower ESG scores are worse
                        compliance_issues.append(
                            f"{ticker} has an ESG score of {total_esg}, which fails to meet ESG requirements."
                        )

                # Check sector restrictions
                if "sector" in restrictions and sector:
                    # Example: checking against restricted sectors
                    restricted_sectors = [
                        "Tobacco",
                        "Defense",
                        "Gambling",
                        "Coal Mining",
                    ]
                    if sector in restricted_sectors or any(
                        s in industry for s in restricted_sectors
                    ):
                        compliance_issues.append(
                            f"{ticker} is in a restricted sector: {sector}/{industry}"
                        )

                # Check if order value exceeds limits
                order_value = quantity * market_price
                if (
                    client_type.lower() == "retail" and order_value > 100000
                ):  # $100,000 for retail
                    compliance_issues.append(
                        f"Order value (${order_value:,.2f}) exceeds retail client threshold of $100,000."
                    )

                # Prepare compliance report
                compliance_result = {
                    "ticker": ticker,
                    "action": action,
                    "quantity": quantity,
                    "market_price": market_price,
                    "order_value": order_value,
                    "client_type": client_type,
                    "sector": sector,
                    "industry": industry,
                    "market_cap": market_cap,
                    "average_daily_volume": average_volume,
                    "restrictions_checked": restrictions,
                    "compliance_status": "Non-compliant"
                    if compliance_issues
                    else "Compliant with warnings"
                    if warnings
                    else "Compliant",
                    "compliance_issues": compliance_issues,
                    "warnings": warnings,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                return json.dumps(compliance_result, indent=2)

            except Exception as e:
                return f"Error while performing compliance check for {ticker}: {str(e)}"

        except Exception as e:
            return f"Could not perform compliance check. Error: {str(e)}"

    async def _arun(self, *args, **kwargs):
        return await asyncio.to_thread(self._run, *args, **kwargs)

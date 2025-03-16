from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Type


class MacroeconomicAnalysisInput(BaseModel):
    indicators: List[str] = Field(
        description="List of macroeconomic indicators to analyze",
        examples=[["GDP", "CPI", "Unemployment"]],
    )
    timeframe: str = Field(
        default="1y",
        description="Timeframe for analysis (e.g., '1y', '6m')",
        examples=["1y"],
    )


class MacroeconomicAnalysisTool(BaseTool):
    name: str = "MacroeconomicAnalysisTool"
    description: str = (
        "Tool for analyzing macroeconomic trends and their potential impact on investments. "
        "Input format: {'indicators': ['GDP', 'CPI', 'Unemployment'], 'timeframe': '1y'} "
        "Available indicators: GDP, CPI, Unemployment, InterestRates, ConsumerSentiment."
    )
    args_schema: Type[BaseModel] = MacroeconomicAnalysisInput

    def _run(self, indicators, timeframe) -> str:
        """Use the tool."""
        try:
            # FRED API base URL (Federal Reserve Economic Data)
            fred_api_key = os.getenv("FRED_API_KEY")
            if not fred_api_key:
                return "FRED API key not found. Using mock data for demonstration."

            # Map of common indicators to their FRED series IDs
            indicator_map = {
                "GDP": "GDP",
                "CPI": "CPIAUCSL",
                "Unemployment": "UNRATE",
                "InterestRates": "FEDFUNDS",
                "ConsumerSentiment": "UMCSENT",
            }

            results = {}

            for indicator in indicators:
                if indicator in indicator_map:
                    series_id = indicator_map[indicator]

                    # In a real implementation, you would make an API call to FRED
                    # For now, we'll generate mock data
                    if fred_api_key != "mock":
                        url = "https://api.stlouisfed.org/fred/series/observations"
                        params = {
                            "series_id": series_id,
                            "api_key": fred_api_key,
                            "file_type": "json",
                            "limit": 12,  # Last 12 observations
                        }
                        response = requests.get(url, params=params)
                        if response.status_code == 200:
                            data = response.json()
                            results[indicator] = data["observations"]
                        else:
                            results[indicator] = (
                                f"Error fetching {indicator} data: {response.status_code}"
                            )
                    else:
                        # Mock data generation
                        dates = [
                            (datetime.now() - timedelta(days=30 * i)).strftime(
                                "%Y-%m-%d"
                            )
                            for i in range(12)
                        ]
                        dates.reverse()

                        if indicator == "GDP":
                            values = [
                                23.32,
                                23.64,
                                24.01,
                                24.20,
                                24.56,
                                24.80,
                                24.65,
                                24.89,
                                25.12,
                                25.45,
                                25.72,
                                26.03,
                            ]
                        elif indicator == "CPI":
                            values = [
                                258.1,
                                258.7,
                                259.4,
                                260.3,
                                261.2,
                                262.5,
                                263.1,
                                264.7,
                                265.2,
                                265.8,
                                266.6,
                                267.0,
                            ]
                        elif indicator == "Unemployment":
                            values = [
                                6.2,
                                6.0,
                                5.8,
                                5.7,
                                5.5,
                                5.4,
                                5.2,
                                5.0,
                                4.9,
                                4.7,
                                4.6,
                                4.5,
                            ]
                        elif indicator == "InterestRates":
                            values = [
                                0.25,
                                0.25,
                                0.25,
                                0.50,
                                0.75,
                                1.00,
                                1.25,
                                1.50,
                                1.75,
                                2.00,
                                2.25,
                                2.50,
                            ]
                        elif indicator == "ConsumerSentiment":
                            values = [
                                76.8,
                                77.3,
                                78.1,
                                77.6,
                                79.0,
                                80.2,
                                81.5,
                                82.3,
                                81.7,
                                81.2,
                                80.6,
                                80.9,
                            ]

                        results[indicator] = [
                            {"date": date, "value": value}
                            for date, value in zip(dates, values)
                        ]
                else:
                    results[indicator] = "Unknown indicator"

            # Add analysis based on the indicators
            analysis = "Macroeconomic Analysis:\n\n"

            if "GDP" in results and isinstance(results["GDP"], list):
                gdp_trend = results["GDP"][-1]["value"] - results["GDP"][0]["value"]
                if gdp_trend > 0:
                    analysis += "- GDP is showing positive growth which generally supports equities.\n"
                else:
                    analysis += "- GDP is contracting which could indicate economic challenges ahead.\n"

            if "CPI" in results and isinstance(results["CPI"], list):
                cpi_change = (
                    (
                        float(results["CPI"][-1]["value"])
                        - float(results["CPI"][-2]["value"])
                    )
                    / float(results["CPI"][-2]["value"])
                    * 100
                )
                analysis += f"- Inflation (CPI) changed by {cpi_change:.2f}% in the latest period.\n"

            if "InterestRates" in results and isinstance(
                results["InterestRates"], list
            ):
                latest_rate = results["InterestRates"][-1]["value"]
                analysis += f"- Current interest rate is {latest_rate}%, which "
                if float(latest_rate) < 1.0:
                    analysis += "is very accommodative for financial markets.\n"
                elif float(latest_rate) < 3.0:
                    analysis += "indicates a neutral monetary policy.\n"
                else:
                    analysis += "suggests a restrictive monetary environment that could pressure growth stocks.\n"

            return json.dumps(
                {"raw_data": results, "analysis": analysis}, indent=2, default=str
            )
        except Exception as e:
            return f"Error analyzing macroeconomic data: {str(e)}"

    async def _arun(self, query: Dict[str, Any] | str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution")

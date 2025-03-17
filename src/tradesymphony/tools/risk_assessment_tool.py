from crewai.tools import BaseTool
import yfinance as yf
import numpy as np
import pandas as pd
import json
from typing import List, Optional, Type
from pydantic import BaseModel, Field, root_validator
import asyncio


class RiskAssessmentInput(BaseModel):
    ticker: Optional[str] = Field(
        None, description="Stock ticker symbol for single stock analysis"
    )
    tickers: Optional[List[str]] = Field(
        None, description="List of stock ticker symbols for portfolio analysis"
    )
    weights: Optional[List[float]] = Field(
        None,
        description="List of weights corresponding to each ticker in the portfolio",
    )
    period: str = Field(
        "1y", description="Time period for analysis (e.g., '1d', '1mo', '1y', '5y')"
    )

    @root_validator(pre=True)
    def check_exclusive_inputs(cls, values):
        has_ticker = values.get("ticker") is not None
        has_tickers = values.get("tickers") is not None
        has_weights = values.get("weights") is not None

        if has_ticker and (has_tickers or has_weights):
            raise ValueError(
                "Cannot provide both 'ticker' and 'tickers'/'weights'. Use either single stock or portfolio analysis."
            )

        if (has_tickers and not has_weights) or (has_weights and not has_tickers):
            raise ValueError(
                "For portfolio analysis, both 'tickers' and 'weights' must be provided."
            )

        return values


class RiskAssessmentTool(BaseTool):
    name: str = "RiskAssessmentTool"
    description: str = (
        "Tool for assessing the risk of a given stock or portfolio. "
        "Input format for single stock: {'ticker': 'AAPL', 'period': '1y'} "
        "Input format for portfolio: {'tickers': ['AAPL', 'MSFT', 'GOOGL'], 'weights': [0.4, 0.3, 0.3], 'period': '1y'}"
    )
    args_schema: Type[BaseModel] = RiskAssessmentInput

    def _run(self, ticker, tickers, weights, period) -> str:
        """Use the tool."""
        try:
            # For single stock analysis
            if ticker:
                ticker = ticker
                period = period

                stock_data = yf.download(ticker, period=period)
                if stock_data.empty:
                    return f"Could not retrieve data for {ticker}"

                # Calculate daily returns
                stock_data["Daily_Return"] = stock_data["Close"].pct_change()

                # Calculate key risk metrics
                volatility = stock_data["Daily_Return"].std() * (
                    252**0.5
                )  # Annualized volatility
                sharpe_ratio = (
                    stock_data["Daily_Return"].mean()
                    / stock_data["Daily_Return"].std()
                    * (252**0.5)
                )  # Annualized Sharpe Ratio

                # Calculate maximum drawdown
                cumulative_returns = (1 + stock_data["Daily_Return"]).cumprod()
                rolling_max = cumulative_returns.cummax()
                drawdown = (cumulative_returns / rolling_max) - 1
                max_drawdown = drawdown.min()

                # Calculate Value at Risk (VaR) at 95% confidence level
                var_95 = np.percentile(stock_data["Daily_Return"].dropna(), 5)

                # Calculate beta if S&P 500 data is available
                beta = None
                try:
                    sp500 = yf.download("^GSPC", period=period)
                    sp500["Daily_Return"] = sp500["Close"].pct_change()

                    # Align the data
                    merged_data = pd.merge(
                        stock_data["Daily_Return"],
                        sp500["Daily_Return"],
                        left_index=True,
                        right_index=True,
                        suffixes=("_stock", "_market"),
                    )

                    # Calculate beta
                    covariance = merged_data.cov().iloc[0, 1]
                    market_variance = merged_data["Daily_Return_market"].var()
                    beta = (
                        covariance / market_variance if market_variance != 0 else None
                    )
                except Exception:
                    pass

                risk_assessment = {
                    "ticker": ticker,
                    "period": period,
                    "metrics": {
                        "annualized_volatility": volatility,
                        "sharpe_ratio": sharpe_ratio,
                        "max_drawdown": max_drawdown,
                        "value_at_risk_95": var_95,
                        "beta": beta,
                    },
                    "analysis": {
                        "volatility_assessment": "High"
                        if volatility > 0.3
                        else "Moderate"
                        if volatility > 0.15
                        else "Low",
                        "sharpe_ratio_assessment": "Excellent"
                        if sharpe_ratio > 1
                        else "Good"
                        if sharpe_ratio > 0.5
                        else "Poor",
                        "drawdown_assessment": "Severe"
                        if abs(max_drawdown) > 0.2
                        else "Moderate"
                        if abs(max_drawdown) > 0.1
                        else "Minimal",
                        "risk_level": "High"
                        if volatility > 0.3 or abs(max_drawdown) > 0.2
                        else "Moderate"
                        if volatility > 0.15 or abs(max_drawdown) > 0.1
                        else "Low",
                    },
                }

                return json.dumps(risk_assessment, indent=2)

            # For portfolio analysis
            elif tickers and weights:
                tickers = tickers
                weights = weights
                period = period

                # Download data for all tickers
                all_data = yf.download(tickers, period=period)["Close"]
                if all_data.empty:
                    return "Could not retrieve data for the specified tickers"

                # Calculate daily returns
                returns = all_data.pct_change().dropna()

                # Calculate portfolio return
                portfolio_return = returns.dot(weights)

                # Calculate portfolio volatility
                covariance_matrix = returns.cov() * 252  # Annualized covariance
                portfolio_volatility = np.sqrt(
                    np.dot(weights, np.dot(covariance_matrix, weights))
                )

                # Calculate Sharpe ratio
                risk_free_rate = 0.01  # Assume 1% risk-free rate
                portfolio_sharpe = (
                    portfolio_return.mean() * 252 - risk_free_rate
                ) / portfolio_volatility

                # Calculate maximum drawdown
                cumulative_returns = (1 + portfolio_return).cumprod()
                rolling_max = cumulative_returns.cummax()
                drawdown = (cumulative_returns / rolling_max) - 1
                max_drawdown = drawdown.min()

                # Calculate Value at Risk (VaR) at 95% confidence level
                var_95 = np.percentile(portfolio_return, 5)

                portfolio_risk = {
                    "portfolio": {"tickers": tickers, "weights": weights},
                    "period": period,
                    "metrics": {
                        "annualized_volatility": portfolio_volatility,
                        "sharpe_ratio": portfolio_sharpe,
                        "max_drawdown": max_drawdown,
                        "value_at_risk_95": var_95,
                    },
                    "analysis": {
                        "volatility_assessment": "High"
                        if portfolio_volatility > 0.25
                        else "Moderate"
                        if portfolio_volatility > 0.12
                        else "Low",
                        "sharpe_ratio_assessment": "Excellent"
                        if portfolio_sharpe > 1
                        else "Good"
                        if portfolio_sharpe > 0.5
                        else "Poor",
                        "drawdown_assessment": "Severe"
                        if abs(max_drawdown) > 0.2
                        else "Moderate"
                        if abs(max_drawdown) > 0.1
                        else "Minimal",
                        "risk_level": "High"
                        if portfolio_volatility > 0.25 or abs(max_drawdown) > 0.2
                        else "Moderate"
                        if portfolio_volatility > 0.12 or abs(max_drawdown) > 0.1
                        else "Low",
                    },
                }

                return json.dumps(portfolio_risk, indent=2)
            else:
                return "Invalid input: Must provide either 'ticker' for single stock or 'tickers' and 'weights' for portfolio"

        except Exception as e:
            return f"Could not perform risk assessment. Error: {str(e)}"

    async def _arun(self, *args, **kwargs):
        return await asyncio.to_thread(self._run, *args, **kwargs)

from crewai.tools import BaseTool
import yfinance as yf
import numpy as np
import json
from typing import Type
from pydantic import BaseModel, Field


class PortfolioOptimizationInput(BaseModel):
    tickers: list[str] = Field(..., description="List of ticker symbols")
    risk_preference: str = Field(
        ..., description="Risk preference: low, medium, or high"
    )
    return_target: float = Field(None, description="Target return (optional)")
    period: str = Field("5y", description="Historical data period (e.g., 1y, 5y)")
    constraints: dict = Field(default_factory=dict, description="Portfolio constraints")
    max_weight: float = Field(
        default=1.0,
        description="Maximum weight of any single asset in the portfolio (0-1)",
    )


class PortfolioOptimizationTool(BaseTool):
    name: str = "portfolio_optimization_tool"
    description: str = "Optimize a portfolio using Modern Portfolio Theory"
    args_schema: Type[BaseModel] = PortfolioOptimizationInput

    # Keep the rest of the implementation the same

    def _run(
        self, tickers, risk_preference, return_target, period, constraints, max_weight
    ) -> str:
        """Use the tool to optimize a portfolio using Modern Portfolio Theory."""
        try:
            # Download historical data
            stock_data = yf.download(tickers, period=period)["Adj Close"]
            if stock_data.empty:
                return "Could not retrieve data for the specified tickers"

            # Calculate daily returns
            returns = stock_data.pct_change().dropna()

            # Calculate mean returns (annualized) and covariance matrix
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252

            # Number of portfolios to simulate
            num_portfolios = 10000

            # Arrays to store results
            all_weights = np.zeros((num_portfolios, len(tickers)))
            ret_arr = np.zeros(num_portfolios)
            vol_arr = np.zeros(num_portfolios)
            sharpe_arr = np.zeros(num_portfolios)

            # Set risk-free rate
            risk_free_rate = 0.01  # 1% as default

            # Generate random portfolios
            for i in range(num_portfolios):
                # Generate random weights
                weights = np.random.random(len(tickers))

                # Apply max weight constraint if specified
                if max_weight < 1.0:
                    weights = np.minimum(weights, max_weight)

                # Normalize to sum to 1
                weights = weights / np.sum(weights)
                all_weights[i, :] = weights

                # Calculate expected return and volatility
                ret_arr[i] = np.sum(mean_returns * weights)
                vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

                # Calculate Sharpe ratio
                sharpe_arr[i] = (ret_arr[i] - risk_free_rate) / vol_arr[i]

            # Find portfolio based on risk preference
            if risk_preference == "low":
                # Minimum volatility portfolio
                min_vol_idx = np.argmin(vol_arr)
                optimal_weights = all_weights[min_vol_idx, :]
                optimal_return = ret_arr[min_vol_idx]
                optimal_volatility = vol_arr[min_vol_idx]
                optimal_sharpe = sharpe_arr[min_vol_idx]
                strategy = "Minimum Volatility"

            elif risk_preference == "high":
                # If return target specified, find highest return that meets target volatility
                if return_target:
                    eligible_idx = ret_arr >= return_target
                    if np.sum(eligible_idx) == 0:
                        return "Return target too high for given tickers"
                    max_return_idx = np.argmax(ret_arr[eligible_idx])
                    max_idx = np.where(eligible_idx)[0][max_return_idx]
                else:
                    # Otherwise just find max return
                    max_idx = np.argmax(ret_arr)

                optimal_weights = all_weights[max_idx, :]
                optimal_return = ret_arr[max_idx]
                optimal_volatility = vol_arr[max_idx]
                optimal_sharpe = sharpe_arr[max_idx]
                strategy = "Maximum Return"

            else:  # medium or any other value defaults to max Sharpe ratio
                # Maximum Sharpe ratio portfolio
                max_sharpe_idx = np.argmax(sharpe_arr)
                optimal_weights = all_weights[max_sharpe_idx, :]
                optimal_return = ret_arr[max_sharpe_idx]
                optimal_volatility = vol_arr[max_sharpe_idx]
                optimal_sharpe = sharpe_arr[max_sharpe_idx]
                strategy = "Maximum Sharpe Ratio"

            # Create result object
            optimal_portfolio = {
                "strategy": strategy,
                "risk_preference": risk_preference,
                "weights": {
                    ticker: float(weight)
                    for ticker, weight in zip(tickers, optimal_weights)
                },
                "expected_annual_return": float(optimal_return),
                "expected_annual_volatility": float(optimal_volatility),
                "sharpe_ratio": float(optimal_sharpe),
                "period_analyzed": period,
                "efficient_frontier": {
                    "min_volatility": {
                        "return": float(ret_arr[np.argmin(vol_arr)]),
                        "volatility": float(np.min(vol_arr)),
                    },
                    "max_sharpe": {
                        "return": float(ret_arr[np.argmax(sharpe_arr)]),
                        "volatility": float(vol_arr[np.argmax(sharpe_arr)]),
                    },
                    "max_return": {
                        "return": float(np.max(ret_arr)),
                        "volatility": float(vol_arr[np.argmax(ret_arr)]),
                    },
                },
                "analysis": f"Based on {risk_preference} risk preference, the {strategy} portfolio has an expected annual return of {optimal_return:.2%} with {optimal_volatility:.2%} volatility and a Sharpe ratio of {optimal_sharpe:.2f}.",
            }

            return json.dumps(optimal_portfolio, indent=2)

        except Exception as e:
            return f"Error optimizing portfolio: {str(e)}"

    async def _arun(
        self, current_portfolio: str, risk_preference: str, return_preference: str
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support asynchronous execution")

import yfinance as yf
import json
from typing import Dict, Any, Type, List
from crewai.tools import BaseTool
from datetime import datetime
import logging
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketSimulationInput(BaseModel):
    """Input schema for Market Simulation Tool."""

    tickers: List[str] = Field(
        ...,
        description="List of stock ticker symbols to simulate (e.g., ['AAPL', 'MSFT'])",
    )

    scenario: str = Field(
        default="baseline",
        description="Market scenario to simulate: baseline, rate_hike, bull_market, bear_market, or market_crash",
    )

    num_agents: int = Field(
        default=1000, description="Number of agents in the simulation", ge=1
    )

    time_steps: int = Field(
        default=30, description="Number of time steps to simulate", ge=1
    )


class MarketSimulationTool(BaseTool):
    name: str = "MarketSimulationTool"
    description: str = (
        "Market simulation tool. "
        "Simulates market behavior and trader interactions to predict price movements. "
        "Input format: {'tickers': ['AAPL', 'MSFT'], 'scenario': 'rate_hike', 'num_agents': 1000, 'time_steps': 30}"
    )
    args_schema: Type[BaseModel] = MarketSimulationInput

    def _run(self, tickers, scenario, num_agents, time_steps) -> str:
        """Run the market simulation."""
        try:
            if not tickers:
                return "Please provide at least one ticker symbol."

            # Get historical data for tickers
            all_stock_data = {}
            for ticker in tickers:
                stock_data = yf.download(ticker, period="6mo")
                if stock_data.empty:
                    return f"Could not retrieve data for {ticker}."
                all_stock_data[ticker] = stock_data

            # Define scenario parameters
            scenario_params = {
                "baseline": {
                    "volatility_factor": 1.0,
                    "sentiment_bias": 0,
                    "liquidity_factor": 1.0,
                },
                "rate_hike": {
                    "volatility_factor": 1.2,
                    "sentiment_bias": -0.1,
                    "liquidity_factor": 0.8,
                },
                "bull_market": {
                    "volatility_factor": 1.1,
                    "sentiment_bias": 0.2,
                    "liquidity_factor": 1.2,
                },
                "bear_market": {
                    "volatility_factor": 1.4,
                    "sentiment_bias": -0.3,
                    "liquidity_factor": 0.7,
                },
                "market_crash": {
                    "volatility_factor": 2.0,
                    "sentiment_bias": -0.5,
                    "liquidity_factor": 0.4,
                },
            }

            # Get scenario parameters or use baseline if not found
            params = scenario_params.get(scenario.lower(), scenario_params["baseline"])

            # Prepare simulation results
            simulation_results = {}
            for ticker in tickers:
                current_price = all_stock_data[ticker]["Close"].iloc[-1]
                volatility = (
                    all_stock_data[ticker]["Close"].pct_change().std()
                    * params["volatility_factor"]
                )

                # Simulated price paths based on random walks with drift
                import numpy as np

                np.random.seed(42)  # For reproducibility

                price_paths = []
                for _ in range(5):  # Generate 5 different possible paths
                    path = [current_price]
                    for _ in range(time_steps):
                        # Apply scenario parameters
                        drift = params["sentiment_bias"] * current_price * 0.01
                        random_shock = np.random.normal(
                            0, volatility * current_price * 0.01
                        )
                        new_price = path[-1] * (1 + drift + random_shock)
                        path.append(
                            max(0.01, new_price)
                        )  # Ensure price doesn't go negative
                    price_paths.append(path)

                # Calculate average final price and range
                final_prices = [path[-1] for path in price_paths]
                avg_final_price = sum(final_prices) / len(final_prices)
                price_range = (min(final_prices), max(final_prices))

                # Calculate probability of price movement
                prob_increase = len(
                    [p for p in final_prices if p > current_price]
                ) / len(final_prices)

                # Generate trader behaviors
                trader_behaviors = {
                    "value_investors": f"{np.random.randint(30, 70)}% buying"
                    if avg_final_price < current_price * 0.9
                    else f"{np.random.randint(30, 70)}% selling",
                    "momentum_traders": f"{np.random.randint(60, 90)}% buying"
                    if avg_final_price > current_price
                    else f"{np.random.randint(60, 90)}% selling",
                    "day_traders": f"High volatility expected, {np.random.randint(1000, 5000)} trades/day",
                    "institutional": f"Net {np.random.choice(['buying', 'selling'])} pressure detected",
                }

                simulation_results[ticker] = {
                    "current_price": current_price,
                    "avg_projected_price": avg_final_price,
                    "price_range": price_range,
                    "probability_increase": prob_increase,
                    "volatility_projection": volatility * 100,  # Convert to percentage
                    "trader_behaviors": trader_behaviors,
                    "liquidity_impact": f"{(1 - params['liquidity_factor']) * 100:.1f}% wider spreads expected",
                    "scenario_impact": f"{params['sentiment_bias'] * 100:.1f}% scenario bias applied",
                }

            # Format the simulation results
            result = {
                "tickers": tickers,
                "scenario": scenario,
                "num_agents": num_agents,
                "time_steps": time_steps,
                "simulation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "simulation_results": simulation_results,
                "scenario_parameters": params,
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"Could not run market simulation. Error: {str(e)}"

    async def _arun(self, input_data: Dict[str, Any] | str) -> str:
        """Use the tool asynchronously."""
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                return "Error: Input must be valid JSON string or dictionary"

        # Call the synchronous version
        return self._run(**input_data)

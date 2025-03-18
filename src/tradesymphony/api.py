from fastapi import FastAPI, HTTPException, BackgroundTasks
import httpx
from typing import Dict, Any
import asyncio
import datetime
import os
import json
from pathlib import Path
from .utils.logger import get_logger

# Import TradeSymphony components
from .crew import InvestmentFirmCrew

app = FastAPI(
    title="TradeSymphony API",
    description="API for running investment analysis using TradeSymphony",
    version="1.0.0",
)

# External API endpoint to fetch portfolio data
PORTFOLIO_API_URL = "https://tradesymphony-client-app.vercel.app/api/trades"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
FALLBACK_DATA_PATH = Path(__file__).parent / "data" / "fallback_portfolio.json"

# Add a logger
logger = get_logger()


@app.get("/")
async def root():
    """Root endpoint that returns basic API information."""
    return {
        "app": "TradeSymphony API",
        "status": "online",
        "endpoints": {
            "/analysis": "POST - Run investment analysis",
            "/health": "GET - Check API health status",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "TradeSymphony API"}


async def fetch_portfolio_data() -> Dict[str, Any]:
    """Fetch portfolio data from the external API with retries and fallback."""
    retries = 0

    while retries < MAX_RETRIES:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.info(
                    f"Attempting to fetch portfolio data from {PORTFOLIO_API_URL}"
                )
                response = await client.get(PORTFOLIO_API_URL)
                response.raise_for_status()
                data = response.json()
                logger.info("Successfully fetched portfolio data")

                # If the API returns a list, wrap it in a dictionary
                if isinstance(data, list):
                    return {"portfolio_items": data}
                return data

        except httpx.TimeoutException as exc:
            logger.error(f"Timeout while fetching portfolio data: {str(exc)}")
            retries += 1
            if retries < MAX_RETRIES:
                logger.info(
                    f"Retrying in {RETRY_DELAY} seconds... (Attempt {retries+1}/{MAX_RETRIES})"
                )
                await asyncio.sleep(RETRY_DELAY)

        except httpx.RequestError as exc:
            logger.error(f"Network error while fetching portfolio data: {str(exc)}")
            retries += 1
            if retries < MAX_RETRIES:
                logger.info(
                    f"Retrying in {RETRY_DELAY} seconds... (Attempt {retries+1}/{MAX_RETRIES})"
                )
                await asyncio.sleep(RETRY_DELAY)

        except httpx.HTTPStatusError as exc:
            logger.error(
                f"Error response from portfolio API: {exc.response.status_code} - {exc.response.text}"
            )
            break  # Don't retry on HTTP status errors

        except Exception as exc:
            logger.error(f"Unexpected error fetching portfolio data: {str(exc)}")
            break

    # If we've exhausted retries or encountered a non-retriable error, use fallback data
    logger.warning("Using fallback portfolio data")
    return await load_fallback_portfolio_data()


async def load_fallback_portfolio_data() -> Dict[str, Any]:
    """Load fallback portfolio data from a local JSON file."""
    try:
        # Ensure directory exists
        os.makedirs(FALLBACK_DATA_PATH.parent, exist_ok=True)

        # If fallback file exists, load it
        if FALLBACK_DATA_PATH.exists():
            with open(FALLBACK_DATA_PATH, "r") as f:
                return json.load(f)

        # Otherwise, return a minimal valid portfolio
        return {
            "portfolio_items": [
                {
                    "symbol": "AAPL",
                    "quantity": 10,
                    "purchase_price": 150.0,
                    "purchase_date": "2023-01-01",
                },
                {
                    "symbol": "MSFT",
                    "quantity": 5,
                    "purchase_price": 300.0,
                    "purchase_date": "2023-01-15",
                },
            ]
        }
    except Exception as exc:
        logger.error(f"Error loading fallback portfolio data: {str(exc)}")
        # Return absolute minimal data that won't break the system
        return {"portfolio_items": []}


async def run_investment_analysis(portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the investment analysis using the InvestmentFirmCrew.

    This runs in a separate thread to avoid blocking the event loop since
    the crew execution might be computationally intensive.
    """
    # Create and kickoff the crew
    try:
        crew = InvestmentFirmCrew(portfolio_data).crew()
        result = crew.kickoff(inputs=portfolio_data)

        # Check if we have structured Pydantic output
        if hasattr(result, "pydantic") and result.pydantic:
            # Convert Pydantic model to dictionary
            return result.pydantic.model_dump()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during investment analysis: {str(e)}"
        )


@app.post("/analysis")
async def analysis(background_tasks: BackgroundTasks):
    """
    Trigger an investment analysis by fetching portfolio data
    and running it through the InvestmentFirmCrew.
    """
    start_time = datetime.datetime.now()

    # Fetch portfolio data
    portfolio_data = await fetch_portfolio_data()

    # Run the analysis in a worker thread to not block the event loop
    # for this potentially long-running task
    result = await asyncio.to_thread(
        lambda: asyncio.run(run_investment_analysis(portfolio_data))
    )

    end_time = datetime.datetime.now()

    return {
        "status": "success",
        "message": "Investment analysis completed",
        "execution_time": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
        },
        "data": result,  # Changed from "results" to "data" to match desired structure
    }


@app.post("/analysis/custom")
async def analysis_with_custom_data(portfolio: Dict[str, Any]):
    """
    Trigger an investment analysis using custom portfolio data provided in the request.
    """
    start_time = datetime.datetime.now()

    result = await asyncio.to_thread(
        lambda: asyncio.run(run_investment_analysis(portfolio))
    )

    end_time = datetime.datetime.now()

    return {
        "status": "success",
        "message": "Custom investment analysis completed",
        "execution_time": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
        },
        "data": result,  # Changed from "results" to "data" to match desired structure
    }


# For direct execution of this file
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

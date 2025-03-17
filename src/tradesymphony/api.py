from fastapi import FastAPI, HTTPException, BackgroundTasks
import httpx
from typing import Dict, Any
import asyncio

# Import TradeSymphony components
from .crew import InvestmentFirmCrew

app = FastAPI(
    title="TradeSymphony API",
    description="API for running investment analysis using TradeSymphony",
    version="1.0.0",
)

# External API endpoint to fetch portfolio data
PORTFOLIO_API_URL = "https://tradesymphony-client-app.vercel.app/api/trades"


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
    """Fetch portfolio data from the external API."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(PORTFOLIO_API_URL)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=503, detail=f"Error fetching portfolio data: {str(exc)}"
            )
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Error response from portfolio API: {exc.response.text}",
            )


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
    # Fetch portfolio data
    portfolio_data = await fetch_portfolio_data()

    # Run the analysis in a worker thread to not block the event loop
    # for this potentially long-running task
    result = await asyncio.to_thread(
        lambda: asyncio.run(run_investment_analysis(portfolio_data))
    )

    return {
        "status": "success",
        "message": "Investment analysis completed",
        "results": result,
    }


@app.post("/analysis/custom")
async def analysis_with_custom_data(portfolio: Dict[str, Any]):
    """
    Trigger an investment analysis using custom portfolio data provided in the request.
    """
    result = await asyncio.to_thread(
        lambda: asyncio.run(run_investment_analysis(portfolio))
    )

    return {
        "status": "success",
        "message": "Custom investment analysis completed",
        "results": result,
    }


# For direct execution of this file
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

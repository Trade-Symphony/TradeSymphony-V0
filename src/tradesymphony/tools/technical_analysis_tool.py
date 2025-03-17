from crewai.tools import BaseTool
import yfinance as yf
import numpy as np
import pandas as pd
import json
from typing import List, Optional, Type
from pydantic import BaseModel, Field
import asyncio


class TechnicalAnalysisInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    indicators: Optional[List[str]] = Field(
        default=["SMA", "RSI", "MACD", "BB"],
        description="Technical indicators to calculate. Can include: SMA, RSI, MACD, BB, ADX",
    )
    period: Optional[str] = Field(
        default="6mo",
        description="Time period for analysis (e.g. '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')",
    )


class TechnicalAnalysisTool(BaseTool):
    name: str = "TechnicalAnalysisTool"
    description: str = (
        "Tool for performing technical analysis on a given stock ticker. "
        "Input format: {'ticker': 'AAPL', 'indicators': ['SMA', 'RSI', 'MACD', 'BB'], 'period': '6mo'}"
    )
    args_schema: Type[BaseModel] = TechnicalAnalysisInput

    def _run(self, ticker, indicators, period) -> str:
        """Use the tool to perform technical analysis on a given stock."""
        try:
            # Download stock data
            stock_data = yf.download(ticker, period=period)
            if stock_data.empty:
                return f"Could not retrieve stock data for {ticker}."

            # Create a dictionary to store analysis results
            analysis_results = {
                "ticker": ticker,
                "period": period,
                "last_price": float(stock_data["Close"].iloc[-1]),
                "date": stock_data.index[-1].strftime("%Y-%m-%d"),
                "indicators": {},
                "signals": [],
            }

            # Calculate indicators and generate signals

            # 1. Moving Averages (SMA)
            if "SMA" in indicators:
                stock_data["SMA_20"] = stock_data["Close"].rolling(window=20).mean()
                stock_data["SMA_50"] = stock_data["Close"].rolling(window=50).mean()
                stock_data["SMA_200"] = stock_data["Close"].rolling(window=200).mean()

                analysis_results["indicators"]["SMA"] = {
                    "SMA_20": float(stock_data["SMA_20"].iloc[-1]),
                    "SMA_50": float(stock_data["SMA_50"].iloc[-1]),
                    "SMA_200": float(stock_data["SMA_200"].iloc[-1]),
                }

                # Generate trading signals for SMA
                current_price = stock_data["Close"].iloc[-1]

                # Golden Cross (bullish): 50-day SMA crosses above 200-day SMA
                golden_cross = (
                    not stock_data["SMA_50"].empty
                    and not stock_data["SMA_200"].empty
                    and stock_data["SMA_50"].iloc[-2] <= stock_data["SMA_200"].iloc[-2]
                    and stock_data["SMA_50"].iloc[-1] > stock_data["SMA_200"].iloc[-1]
                )

                # Death Cross (bearish): 50-day SMA crosses below 200-day SMA
                death_cross = (
                    not stock_data["SMA_50"].empty
                    and not stock_data["SMA_200"].empty
                    and stock_data["SMA_50"].iloc[-2] >= stock_data["SMA_200"].iloc[-2]
                    and stock_data["SMA_50"].iloc[-1] < stock_data["SMA_200"].iloc[-1]
                )

                if golden_cross:
                    analysis_results["signals"].append(
                        {
                            "indicator": "SMA",
                            "signal": "BUY",
                            "strength": "STRONG",
                            "description": "Golden Cross: 50-day SMA crossed above 200-day SMA, indicating potential bullish trend.",
                        }
                    )
                elif death_cross:
                    analysis_results["signals"].append(
                        {
                            "indicator": "SMA",
                            "signal": "SELL",
                            "strength": "STRONG",
                            "description": "Death Cross: 50-day SMA crossed below 200-day SMA, indicating potential bearish trend.",
                        }
                    )
                elif (
                    current_price
                    > stock_data["SMA_50"].iloc[-1]
                    > stock_data["SMA_200"].iloc[-1]
                ):
                    analysis_results["signals"].append(
                        {
                            "indicator": "SMA",
                            "signal": "BUY",
                            "strength": "MODERATE",
                            "description": "Price above 50-day and 200-day SMAs, indicating bullish trend.",
                        }
                    )
                elif (
                    current_price
                    < stock_data["SMA_50"].iloc[-1]
                    < stock_data["SMA_200"].iloc[-1]
                ):
                    analysis_results["signals"].append(
                        {
                            "indicator": "SMA",
                            "signal": "SELL",
                            "strength": "MODERATE",
                            "description": "Price below 50-day and 200-day SMAs, indicating bearish trend.",
                        }
                    )

            # 2. Relative Strength Index (RSI)
            if "RSI" in indicators:
                # Calculate RSI using 14-day period
                delta = stock_data["Close"].diff()
                gain = delta.where(delta > 0, 0).fillna(0)
                loss = -delta.where(delta < 0, 0).fillna(0)

                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()

                rs = avg_gain / avg_loss
                stock_data["RSI"] = 100 - (100 / (1 + rs))

                current_rsi = float(stock_data["RSI"].iloc[-1])
                analysis_results["indicators"]["RSI"] = {
                    "value": current_rsi,
                    "interpretation": "overbought"
                    if current_rsi > 70
                    else "oversold"
                    if current_rsi < 30
                    else "neutral",
                }

                # Generate trading signals for RSI
                if current_rsi > 70:
                    analysis_results["signals"].append(
                        {
                            "indicator": "RSI",
                            "signal": "SELL",
                            "strength": "STRONG" if current_rsi > 80 else "MODERATE",
                            "description": f"RSI at {current_rsi:.2f}, indicating overbought conditions.",
                        }
                    )
                elif current_rsi < 30:
                    analysis_results["signals"].append(
                        {
                            "indicator": "RSI",
                            "signal": "BUY",
                            "strength": "STRONG" if current_rsi < 20 else "MODERATE",
                            "description": f"RSI at {current_rsi:.2f}, indicating oversold conditions.",
                        }
                    )

            # 3. Moving Average Convergence Divergence (MACD)
            if "MACD" in indicators:
                # Calculate MACD
                stock_data["EMA_12"] = (
                    stock_data["Close"].ewm(span=12, adjust=False).mean()
                )
                stock_data["EMA_26"] = (
                    stock_data["Close"].ewm(span=26, adjust=False).mean()
                )
                stock_data["MACD"] = stock_data["EMA_12"] - stock_data["EMA_26"]
                stock_data["Signal_Line"] = (
                    stock_data["MACD"].ewm(span=9, adjust=False).mean()
                )
                stock_data["MACD_Histogram"] = (
                    stock_data["MACD"] - stock_data["Signal_Line"]
                )

                current_macd = float(stock_data["MACD"].iloc[-1])
                current_signal = float(stock_data["Signal_Line"].iloc[-1])
                current_histogram = float(stock_data["MACD_Histogram"].iloc[-1])

                analysis_results["indicators"]["MACD"] = {
                    "MACD": current_macd,
                    "Signal_Line": current_signal,
                    "Histogram": current_histogram,
                }

                # Generate trading signals for MACD
                # Bullish crossover
                if (
                    not stock_data["MACD"].empty
                    and not stock_data["Signal_Line"].empty
                    and stock_data["MACD"].iloc[-2] < stock_data["Signal_Line"].iloc[-2]
                    and stock_data["MACD"].iloc[-1] > stock_data["Signal_Line"].iloc[-1]
                ):
                    analysis_results["signals"].append(
                        {
                            "indicator": "MACD",
                            "signal": "BUY",
                            "strength": "STRONG",
                            "description": "MACD crossed above signal line, indicating bullish momentum.",
                        }
                    )
                # Bearish crossover
                elif (
                    not stock_data["MACD"].empty
                    and not stock_data["Signal_Line"].empty
                    and stock_data["MACD"].iloc[-2] > stock_data["Signal_Line"].iloc[-2]
                    and stock_data["MACD"].iloc[-1] < stock_data["Signal_Line"].iloc[-1]
                ):
                    analysis_results["signals"].append(
                        {
                            "indicator": "MACD",
                            "signal": "SELL",
                            "strength": "STRONG",
                            "description": "MACD crossed below signal line, indicating bearish momentum.",
                        }
                    )
                # MACD above signal line
                elif current_macd > current_signal:
                    analysis_results["signals"].append(
                        {
                            "indicator": "MACD",
                            "signal": "BUY",
                            "strength": "MODERATE",
                            "description": "MACD above signal line, indicating bullish momentum.",
                        }
                    )
                # MACD below signal line
                elif current_macd < current_signal:
                    analysis_results["signals"].append(
                        {
                            "indicator": "MACD",
                            "signal": "SELL",
                            "strength": "MODERATE",
                            "description": "MACD below signal line, indicating bearish momentum.",
                        }
                    )

            # 4. Bollinger Bands (BB)
            if "BB" in indicators:
                # Calculate Bollinger Bands
                window = 20
                std_dev = 2

                stock_data["BB_Middle"] = (
                    stock_data["Close"].rolling(window=window).mean()
                )
                rolling_std = stock_data["Close"].rolling(window=window).std()
                stock_data["BB_Upper"] = stock_data["BB_Middle"] + (
                    rolling_std * std_dev
                )
                stock_data["BB_Lower"] = stock_data["BB_Middle"] - (
                    rolling_std * std_dev
                )

                current_price = float(stock_data["Close"].iloc[-1])
                current_upper = float(stock_data["BB_Upper"].iloc[-1])
                current_middle = float(stock_data["BB_Middle"].iloc[-1])
                current_lower = float(stock_data["BB_Lower"].iloc[-1])
                # Calculate Bollinger Band width (volatility indicator)
                bb_width = (current_upper - current_lower) / current_middle

                analysis_results["indicators"]["BollingerBands"] = {
                    "Upper": current_upper,
                    "Middle": current_middle,
                    "Lower": current_lower,
                    "Width": float(bb_width),
                    "PercentB": float(
                        (current_price - current_lower)
                        / (current_upper - current_lower)
                    )
                    if (current_upper - current_lower) > 0
                    else 0,
                }

                # Generate trading signals for Bollinger Bands
                if current_price > current_upper:
                    analysis_results["signals"].append(
                        {
                            "indicator": "BB",
                            "signal": "SELL",
                            "strength": "MODERATE",
                            "description": "Price above upper Bollinger Band, indicating overbought conditions or strong uptrend.",
                        }
                    )
                elif current_price < current_lower:
                    analysis_results["signals"].append(
                        {
                            "indicator": "BB",
                            "signal": "BUY",
                            "strength": "MODERATE",
                            "description": "Price below lower Bollinger Band, indicating oversold conditions or strong downtrend.",
                        }
                    )

            # 5. Average Directional Index (ADX) - Optional
            if "ADX" in indicators and len(stock_data) > 14:
                # Calculate ADX
                high_low = stock_data["High"] - stock_data["Low"]
                high_close = abs(stock_data["High"] - stock_data["Close"].shift())
                low_close = abs(stock_data["Low"] - stock_data["Close"].shift())

                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean()

                up_move = stock_data["High"] - stock_data["High"].shift()
                down_move = stock_data["Low"].shift() - stock_data["Low"]

                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                minus_dm = np.where(
                    (down_move > up_move) & (down_move > 0), down_move, 0
                )

                plus_di = 100 * pd.Series(plus_dm).rolling(window=14).mean() / atr
                minus_di = 100 * pd.Series(minus_dm).rolling(window=14).mean() / atr

                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                adx = dx.rolling(window=14).mean()

                stock_data["ADX"] = adx
                stock_data["Plus_DI"] = plus_di
                stock_data["Minus_DI"] = minus_di

                current_adx = float(stock_data["ADX"].iloc[-1])
                current_plus_di = float(stock_data["Plus_DI"].iloc[-1])
                current_minus_di = float(stock_data["Minus_DI"].iloc[-1])

                trend_strength = (
                    "weak"
                    if current_adx < 25
                    else "moderate"
                    if current_adx < 50
                    else "strong"
                    if current_adx < 75
                    else "extreme"
                )

                analysis_results["indicators"]["ADX"] = {
                    "ADX": current_adx,
                    "Plus_DI": current_plus_di,
                    "Minus_DI": current_minus_di,
                    "Trend_Strength": trend_strength,
                }

                # Generate trading signals for ADX
                if current_adx > 25:
                    if current_plus_di > current_minus_di:
                        analysis_results["signals"].append(
                            {
                                "indicator": "ADX",
                                "signal": "BUY",
                                "strength": "MODERATE"
                                if current_adx < 50
                                else "STRONG",
                                "description": f"ADX at {current_adx:.2f} with +DI above -DI, indicating strong uptrend.",
                            }
                        )
                    elif current_minus_di > current_plus_di:
                        analysis_results["signals"].append(
                            {
                                "indicator": "ADX",
                                "signal": "SELL",
                                "strength": "MODERATE"
                                if current_adx < 50
                                else "STRONG",
                                "description": f"ADX at {current_adx:.2f} with -DI above +DI, indicating strong downtrend.",
                            }
                        )

            # Generate overall recommendation based on signals
            buy_signals = [
                s for s in analysis_results["signals"] if s["signal"] == "BUY"
            ]
            sell_signals = [
                s for s in analysis_results["signals"] if s["signal"] == "SELL"
            ]

            strong_buy = len([s for s in buy_signals if s["strength"] == "STRONG"])
            strong_sell = len([s for s in sell_signals if s["strength"] == "STRONG"])
            moderate_buy = len([s for s in buy_signals if s["strength"] == "MODERATE"])
            moderate_sell = len(
                [s for s in sell_signals if s["strength"] == "MODERATE"]
            )

            total_buy_strength = strong_buy * 2 + moderate_buy
            total_sell_strength = strong_sell * 2 + moderate_sell

            if total_buy_strength > total_sell_strength * 2:
                recommendation = "STRONG BUY"
            elif total_buy_strength > total_sell_strength:
                recommendation = "BUY"
            elif total_sell_strength > total_buy_strength * 2:
                recommendation = "STRONG SELL"
            elif total_sell_strength > total_buy_strength:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"

            analysis_results["recommendation"] = recommendation
            analysis_results["summary"] = (
                f"Technical analysis for {ticker} indicates a {recommendation} recommendation with {len(buy_signals)} buy signals and {len(sell_signals)} sell signals."
            )

            return json.dumps(analysis_results, indent=2)

        except Exception as e:
            return f"Could not perform technical analysis for. Error: {str(e)}"

    async def _arun(self, *args, **kwargs):
        return await asyncio.to_thread(self._run, *args, **kwargs)

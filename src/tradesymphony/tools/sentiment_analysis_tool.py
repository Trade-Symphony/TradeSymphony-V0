from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os
import json
import requests
from typing import Type
import asyncio


class SentimentAnalysisInput(BaseModel):
    company: str = Field(
        description="The company name or ticker symbol to analyze sentiment for"
    )


class SentimentAnalysisTool(BaseTool):
    name: str = "SentimentAnalysisTool"
    description: str = (
        "Tool for performing sentiment analysis on news articles or social media related to a company. "
        "Input should be the company name or ticker symbol."
    )
    args_schema: Type[BaseModel] = SentimentAnalysisInput

    def _run(self, company: str) -> str:
        """Use the tool."""
        try:
            # First, search for recent news about the company
            serper_api_key = os.getenv("SERPER_API_KEY")
            if not serper_api_key:
                return "Google Serper API key not found in environment variables."

            # Search for recent news
            headers = {
                "X-API-KEY": serper_api_key,
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",  # Add User-Agent
            }
            payload = json.dumps(
                {
                    "q": f"{company} stock news",
                    "gl": "us",
                    "hl": "en",
                    "num": 10,
                    "search_type": "news",
                }
            )

            response = requests.post(
                "https://google.serper.dev/search", headers=headers, data=payload
            )

            if response.status_code != 200:
                return f"Failed to fetch news data: {response.status_code}"

            search_results = response.json()
            news_items = search_results.get("news", [])

            if not news_items:
                return f"No recent news found for {company}"

            # Now analyze sentiment for each news item
            from textblob import TextBlob
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            sentiment_results = []

            overall_polarity = 0
            overall_subjectivity = 0
            overall_compound = 0

            for item in news_items[:5]:  # Analyze top 5 news items
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                published_date = item.get("date", "Unknown")
                source = item.get("source", "Unknown")
                link = item.get("link", "")

                # TextBlob sentiment (polarity: -1 to 1, subjectivity: 0 to 1)
                title_blob = TextBlob(title)
                snippet_blob = TextBlob(snippet)
                polarity = (
                    title_blob.sentiment.polarity + snippet_blob.sentiment.polarity
                ) / 2
                subjectivity = (
                    title_blob.sentiment.subjectivity
                    + snippet_blob.sentiment.subjectivity
                ) / 2

                # VADER sentiment (compound score: -1 to 1)
                title_scores = analyzer.polarity_scores(title)
                snippet_scores = analyzer.polarity_scores(snippet)
                compound = (title_scores["compound"] + snippet_scores["compound"]) / 2

                # Accumulate for overall sentiment
                overall_polarity += polarity
                overall_subjectivity += subjectivity
                overall_compound += compound

                sentiment_results.append(
                    {
                        "title": title,
                        "source": source,
                        "date": published_date,
                        "link": link,
                        "sentiment": {
                            "polarity": polarity,
                            "subjectivity": subjectivity,
                            "compound": compound,
                            "classification": "positive"
                            if compound > 0.05
                            else "negative"
                            if compound < -0.05
                            else "neutral",
                        },
                    }
                )

            # Calculate overall sentiment
            if news_items:
                overall_polarity /= len(news_items[:5])
                overall_subjectivity /= len(news_items[:5])
                overall_compound /= len(news_items[:5])

            # Interpret overall sentiment
            sentiment_classification = (
                "positive"
                if overall_compound > 0.05
                else "negative"
                if overall_compound < -0.05
                else "neutral"
            )
            sentiment_strength = (
                "strong"
                if abs(overall_compound) > 0.5
                else "moderate"
                if abs(overall_compound) > 0.2
                else "weak"
            )

            result = {
                "company": company,
                "news_items": sentiment_results,
                "overall_sentiment": {
                    "polarity": overall_polarity,
                    "subjectivity": overall_subjectivity,
                    "compound": overall_compound,
                    "classification": sentiment_classification,
                    "strength": sentiment_strength,
                },
                "analysis": f"The overall sentiment for {company} is {sentiment_classification} with {sentiment_strength} intensity. "
                f"News coverage has a subjectivity score of {overall_subjectivity:.2f}, indicating "
                f"{'highly subjective' if overall_subjectivity > 0.7 else 'moderately subjective' if overall_subjectivity > 0.4 else 'relatively objective'} reporting.",
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return (
                f"Could not perform sentiment analysis for {company}. Error: {str(e)}"
            )

    async def _arun(self, *args, **kwargs):
        return await asyncio.to_thread(self._run, *args, **kwargs)

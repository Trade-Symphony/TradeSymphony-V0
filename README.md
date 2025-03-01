# TradeSymphony: The Multi-Agentic Stock Trading System

## Overview

TradeSymphony is a multi-agent trading simulation system where independent agents represent traders interacting with a simulated stock market. The system allows for the setup of external conditions to observe how market dynamics are affected by various factors.

## Use Case Description

TradeSymphony creates a realistic trading environment where multiple AI agents collaborate to analyze market data, make trading decisions, and optimize performance over time. The system helps users understand complex market dynamics, test trading strategies in a risk-free environment, and develop insights into how different factors influence stock market behavior.

## Technical Specifications

- Scraper Agent
  - Responsible for gathering raw market data by integrating with external data providers such as the Yahoo Finance API, various financial news and data APIs, and even performing automated Google searches (with queries generated automatically).
  - Collects real-time information from financial reports, stock prices, market trends, and news on economics, finance, and politics.
  - Operates on a time-triggered basis (every few minutes) and initiates the Neutralization Agent once data is collected.
- Neutralization Agent
  - Processes and cleans the collected data using an Evaluator-Optimizer workflow that leverages a wide range of APIs (e.g., Biaslyze API) to ensure the text meets the required level of neutralization.
  - Separates facts from opinions and standardizes the data for consistent downstream processing.
  - Triggers the Market Analysis Agent after data normalization.
- Market Analysis Agent
  - Extracts and refines the most important details from the normalized data using integrated Technical, Fundamental, and Sentiment Analysis components.
  - Stores processed information in a high-speed Quadrant Vector Database for quick access and in MongoDB for robust, long-term storage.
  - Components include:
    - Technical Analysis: Studies price patterns, trends, and momentum indicators
    - Fundamental Analysis: Evaluates company financials, earnings, and growth metrics
    - Market Sentiment: Analyzes investor behavior and emotions
- Trading Agent
  - Leverages insights derived from the Market Analysis Agent to interact with a dedicated Trading API.
  - Responsible for executing trades based on real-time market conditions and runs continuously (24/7 or as defined by the trading timeframe).
  - Makes decisions based on comprehensive market analysis, risk management principles, and trading strategy parameters.
- Feedback Agent
  - Monitors executed trades and overall system performance by integrating with prompt optimization mechanisms.
  - Employs an Evaluator-Optimizer architecture alongside an Autonomous Agent framework (as exemplified by Agent Recipes).
  - Continuously refines trading decisions and optimizes system operations.

### External APIs and Systems

- Yahoo Finance API (and similar financial data APIs) for pulling stock data and financial reports.
- News API (e.g., NewsAPI.org) for gathering financial news.
- Bloomberg Open API for additional market data.
- Biaslyze API (e.g., https://biaslyze.org/api/) for evaluating and optimizing text neutrality.
- Trading API for executing trades.

## Agentic Flow Diagram

The TradeSymphony architecture consists of several interconnected components as illustrated in the `Agentic-Flow-Diagram.mmd`

### Data Flow

1. Data sources provide raw information to preprocessing agents
2. Preprocessing agents clean and prepare data for analysis
3. Market analysis components evaluate data using different methodologies
4. Trading agents make decisions based on analyzed data
5. Feedback loops optimize the system performance through continuous learning

## Database Integration

The application leverages multiple databases to manage and store data effectively:

### MongoDB

- Acts as the primary storage solution for historical market data, agent decisions, and performance metrics.
- Provides robust, long-term data storage, ensuring data integrity and scalability for extensive archival and analytical purposes.

### QDrant Vector DB

- Facilitates advanced data analytics and high-dimensional similarity searches, which are critical for nuanced sentiment analysis and technical evaluations.
- Dual Storage Architecture: QDrant is designed with a two-layer storage approach:
  - Short-Term Storage: An in-memory section optimized for fast, real-time data access and rapid query performance.
  - Long-Term Storage: A persistent, on-disk (memmap) section that securely archives historical vector data, ensuring scalability and cost-effective storage.
- This dual approach enables QDrant to deliver immediate, low-latency insights while maintaining a comprehensive repository for long-term analysis.

## Memory Framework

Our system employs a dual-layer memory framework that supports both short-term and long-term memory, ensuring that our agents can react promptly to current market conditions while also learning from historical data to improve future decision-making.

### Short-Term Memory

Utilizes in-memory caches to quickly adapt to current market conditions.
Provides immediate feedback from trade executions, enabling rapid responses to volatile market movements.
Integrated QDrant Vector DB: In addition to traditional in-memory caching, we integrate QDrant into our short-term memory layer. QDrant's high-speed, in-memory vector search capabilities allow us to quickly retrieve and compare relevant data points, enhancing real-time decision-making by agents such as the Trading Agent.

### Long-Term Memory

Leverages persistent databases like MongoDB to store historical market data, trade logs, and performance analytics.
Enables the analysis of past market behaviours, allowing agents to learn from historical patterns.
Employs QDrant's on-disk (memmap) storage option to archive vector data efficiently, ensuring scalability and cost-effective long-term data management.

### Business Value

This dual-layer memory architecture not only enhances our system's responsiveness to immediate market dynamics but also builds a robust knowledge base for future strategy refinement. By combining rapid in-memory caches and integrated QDrant vector search for short-term needs with comprehensive, persistent storage for long-term analysis, our agents are empowered to make more informed, optimized trading decisions—ultimately driving better performance and profitability.

## Decision Factors

### Macroeconomic Factors

- Interest rates and monetary policy
- Inflation rates and expectations
- GDP growth and economic indicators
- Employment data and wage growth
- Global economic conditions and trade relations

### Market Analysis Metrics

- Technical analysis: Price patterns, trends, and momentum indicators
- Fundamental analysis: Company financials, earnings, P/E ratios, growth metrics
- Market sentiment and psychology: Investor behavior and emotions

### Company-Specific Factors

- Financial statements and performance metrics
- Management team quality and strategy
- Competitive position in the industry
- Recent news and corporate events
- Regulatory environment and compliance

### Risk Management

- Position sizing and portfolio diversification
- Stop-loss levels and risk-reward ratios
- Market volatility and liquidity conditions
- Margin requirements and leverage exposure
- Correlation with other assets

### Trading Costs and Logistics

- Commission fees and transaction costs
- Bid-ask spreads and slippage
- Tax implications of trades
- Available trading capital
- Time horizon for investment

### Market Timing

- Current market phase (bull/bear market)
- Sector rotation and industry trends
- Seasonal patterns and market cycles
- Trading volume and participation
- Upcoming economic events or earnings releases

## Business Value

TradeSymphony delivers significant value through:

1. Risk-Free Strategy Testing: Test complex trading strategies without risking actual capital
2. Data-Driven Insights: Gain deep market insights through multi-faceted analysis
3. Automated Trading: Reduce manual intervention through AI-powered decision making
4. Continuous Learning: System adapts and improves based on market performance
5. Reduced Emotional Bias: Eliminate human emotional bias from trading decisions

## Optional Feature(s)

### Prompt Auto-Optimizer

The Prompt Auto-Optimizer is an advanced component that leverages real-time feedback from trade outcomes to continuously refine input prompts and adjust model parameters. By applying machine learning techniques—including reinforcement learning and adaptive algorithms—it dynamically tunes agent inputs to enhance decision accuracy and accelerate response times.

#### Key Capabilities

- Real-Time Feedback Integration: Monitors trade performance and market dynamics to identify areas for improvement in prompt design.
- Dynamic Parameter Adjustment: Automatically refines model parameters and prompt structures based on evolving data, ensuring that agents are always working with the most effective inputs.
- Adaptive Learning: Utilizes advanced ML algorithms to learn from historical trends (stored in long-term memory via MongoDB and QDrant's on-disk storage) as well as immediate market conditions (captured in in-memory caches and QDrant's short-term storage) to drive prompt optimization.
- Seamless Integration: Works harmoniously with our dual-layer memory framework—leveraging both high-speed in-memory caches (enhanced by QDrant for rapid vector search) and persistent data stores—to ensure that optimization is both context-aware and scalable.
- Improved Decision-Making: By continually fine-tuning the inputs fed to our AI agents, the optimizer boosts trade execution precision, reduces latency in responses, and ultimately drives better trading performance.

## Technical Implementation

### Development Framework

- MAS Framework: Langraph for agent workflow orchestration
- Vector Database Integration: Qdrant for similarity search and retrieval
- Persistent Storage: MongoDB for structured data

## Feedback and Continuous Improvement

The system incorporates feedback loops for continuous improvement:

- Prompt refinement for agent interactions
- Performance analysis and strategy optimization
- Adaptive learning based on market conditions

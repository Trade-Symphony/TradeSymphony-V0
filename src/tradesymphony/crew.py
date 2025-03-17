from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from langchain.callbacks import LangChainTracer
from langsmith import Client, traceable
import langsmith
from typing import Dict, Any
from .models.investment_recommendation import InvestmentRecommendation

import os
from dotenv import load_dotenv
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.tasks import TaskOutput
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

# Import custom tools
from .tools import (
    StockScreenerTool,
    FinancialDataTool,
    MacroeconomicAnalysisTool,
    SentimentAnalysisTool,
    RiskAssessmentTool,
    ComplianceCheckTool,
    PortfolioOptimizationTool,
    TechnicalAnalysisTool,
    TavilySearchTool,
    FirecrawlResearchTool,
    CompanyResearchTool,
    AlphaVantageTool,
    YFinanceTool,
    get_firecrawl_crawl_website_tool,
    get_firecrawl_scrape_website_tool,
    StockSymbolFetcherTool,
)

load_dotenv()


@CrewBase
class InvestmentFirmCrew:
    """
    Hierarchical Investment Firm Crew that simulates the structure and workflow
    of a professional investment management firm.

    This crew follows industry standard practices with a hierarchy of specialized
    agents working together to analyze portfolios and make investment recommendations.
    """

    # Configuration files
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # Knowledge sources
    investment_structure_knowledge = (
        "knowledge/Structure of the Investment Industry.pdf"
    )
    # market_knowledge = "knowledge/market_data.json"

    def __init__(self, portfolio_input: Dict[str, Any], verbose: bool = True):
        """
        Initialize the Investment Firm Crew with a portfolio input.

        Args:
            portfolio_input: Dictionary containing portfolio details
            verbose: Whether to enable verbose output
        """
        self.portfolio_input = portfolio_input
        self.verbose = verbose
        self.langsmith_client = Client(
            api_key=os.getenv("LANGCHAIN_API_KEY"),
        )
        self.tracer = LangChainTracer(
            project_name=os.getenv("LANGCHAIN_PROJECT", "tradesymphony")
        )

    @before_kickoff
    def setup_environment(self, inputs):
        """Initialize the environment before running the crew."""
        try:
            self.langsmith_client.create_project(
                os.getenv("LANGCHAIN_PROJECT", "tradesymphony"),
                upsert=True,  # Use upsert to avoid conflicts
            )
        except langsmith.utils.LangSmithConflictError:
            # Session already exists, this is fine
            print("ℹ️ LangSmith session already exists, continuing...")
        except Exception as e:
            # Log other errors but don't fail the pipeline
            print(f"⚠️ LangSmith error (non-critical): {e}")

        # Process the inputs
        print(f"📊 Analyzing portfolio: {inputs.get('portfolio_name', 'Unknown')}")
        return inputs

    @after_kickoff
    def finalize_operations(self, result):
        """Perform cleanup after the crew completes operation."""
        print("✅ Investment analysis completed")
        print(
            f"📑 Results summary: {len(result.get('recommendations', []))} investment recommendations generated"
        )
        return result

    @agent
    @traceable
    def crew_manager(self) -> Agent:
        """
        Crew Manager is responsible for coordinating the workflow between agents,
        managing the execution sequence, and ensuring collaboration effectiveness.
        This agent doesn't use tools but focuses on process management.
        """
        return Agent(
            config={
                "role": "Investment Firm Coordinator",
                "goal": "Effectively coordinate the investment analysis workflow and ensure optimal collaboration between specialized agents",
                "backstory": "You are an experienced investment firm coordinator with exceptional organizational skills. You understand the investment process deeply and know how to efficiently manage complex workflows involving multiple specialized professionals. Your focus is on ensuring clear communication, proper task delegation, and synthesizing insights from various team members into coherent outcomes.",
            },
            tools=[],  # No tools for the manager, purely coordination-focused
            verbose=self.verbose,
            allow_delegation=True,
            memory=True,  # Needs memory to track the entire process
            respect_context_window=True,
            cache=True,
            # max_rpm=5,
        )

    # C-Suite Level Agents
    @traceable
    @agent
    def chief_investment_officer(self) -> Agent:
        """
        The Chief Investment Officer (CIO) is responsible for overall investment
        strategy and making final portfolio decisions.
        """
        financial_tools = [
            PortfolioOptimizationTool(),
            MacroeconomicAnalysisTool(),
            RiskAssessmentTool(),
            TavilySearchTool(),
            FinancialDataTool(),
            SentimentAnalysisTool(),
            StockSymbolFetcherTool(),
        ]
        return Agent(
            config=self.agents_config["chief_investment_officer"],
            tools=financial_tools,
            verbose=self.verbose,
            allow_delegation=True,
            # max_iter=5,
            memory=True,  # Enable memory for context retention
            respect_context_window=True,
            max_rpm=10,  # Rate limiting to avoid API throttling
            cache=True,  # Cache responses for identical requests
            max_retry_limit=3,  # Increase retries for this critical agent
        )

    @traceable
    @agent
    def investment_committee(self) -> Agent:
        """
        The Investment Committee reviews and votes on all major investment
        decisions before final implementation.
        """
        committee_tools = [
            RiskAssessmentTool(),
            PortfolioOptimizationTool(),
            ComplianceCheckTool(),
            FinancialDataTool(),
            StockSymbolFetcherTool(),
        ]
        return Agent(
            config=self.agents_config["investment_committee"],
            tools=committee_tools,
            verbose=self.verbose,
            allow_delegation=False,
            memory=True,  # Committee needs to remember previous discussions
            respect_context_window=True,
            # max_rpm=8,
            cache=True,
        )

    @traceable
    @agent
    def chief_compliance_officer(self) -> Agent:
        """
        The Chief Compliance Officer ensures all investment decisions
        comply with regulations and internal policies.
        """
        compliance_tools = [
            ComplianceCheckTool(),
            TavilySearchTool(),
            FirecrawlResearchTool(),
            SentimentAnalysisTool(),
            StockSymbolFetcherTool(),
        ]
        return Agent(
            config=self.agents_config["chief_compliance_officer"],
            tools=compliance_tools,
            verbose=self.verbose,
            allow_delegation=False,
            memory=True,  # Compliance needs memory for consistent rulings
            respect_context_window=True,
            # max_rpm=5,
            cache=True,
        )

    @traceable
    @agent
    def portfolio_manager(self) -> Agent:
        """
        Portfolio Manager is responsible for executing investment strategies
        and managing specific portfolios.
        """
        portfolio_tools = [
            PortfolioOptimizationTool(),
            StockScreenerTool(),
            FinancialDataTool(),
            TechnicalAnalysisTool(),
            RiskAssessmentTool(),
            TavilySearchTool(),
            StockSymbolFetcherTool(),
        ]
        return Agent(
            config=self.agents_config["portfolio_manager"],
            tools=portfolio_tools,
            verbose=self.verbose,
            allow_delegation=True,
            memory=True,
            respect_context_window=True,
            # max_rpm=10,
            cache=True,
        )

    @traceable
    @agent
    def risk_manager(self) -> Agent:
        """
        Risk Manager evaluates and mitigates risks in investment portfolios.
        """
        risk_tools = [
            RiskAssessmentTool(),
            PortfolioOptimizationTool(),
            MacroeconomicAnalysisTool(),
            TechnicalAnalysisTool(),
            FinancialDataTool(),
            StockSymbolFetcherTool(),
        ]
        return Agent(
            config=self.agents_config["risk_manager"],
            tools=risk_tools,
            verbose=self.verbose,
            allow_delegation=False,
            memory=True,  # Risk management requires historical context
            respect_context_window=True,
            # max_rpm=8,
            cache=True,
        )

    @traceable
    @agent
    def fundamental_research_analyst(self) -> Agent:
        """
        Fundamental Research Analyst performs detailed company and industry analysis.
        """
        research_tools = [
            TavilySearchTool(),
            FirecrawlResearchTool(),
            CompanyResearchTool(),
            FinancialDataTool(),
            YFinanceTool(),
            AlphaVantageTool(),
            MacroeconomicAnalysisTool(),
            get_firecrawl_crawl_website_tool(),
            get_firecrawl_scrape_website_tool(),
            StockSymbolFetcherTool(),
        ]

        return Agent(
            config=self.agents_config["fundamental_research_analyst"],
            tools=research_tools,
            verbose=self.verbose,
            allow_delegation=False,
            memory=True,  # Research requires context from previous findings
            respect_context_window=True,
            # max_rpm=15,  # Higher RPM for research-intensive role
            cache=True,
        )

    @traceable
    @agent
    def quantitative_analyst(self) -> Agent:
        """
        Quantitative Analyst applies mathematical and statistical techniques
        to analyze investment opportunities and risks.
        """
        quant_tools = [
            StockScreenerTool(),
            FinancialDataTool(),
            TechnicalAnalysisTool(),
            YFinanceTool(),
            AlphaVantageTool(),
            PortfolioOptimizationTool(),
            RiskAssessmentTool(),
            StockSymbolFetcherTool(),
        ]
        return Agent(
            config=self.agents_config["quantitative_analyst"],
            tools=quant_tools,
            verbose=self.verbose,
            allow_delegation=False,
            memory=True,
            respect_context_window=True,
            # max_rpm=15,  # Higher RPM for data-intensive operations
            cache=True,
        )

    @traceable
    @agent
    def esg_analyst(self) -> Agent:
        """
        ESG Analyst assesses environmental, social, and governance factors.
        """
        esg_tools = [
            TavilySearchTool(),
            CompanyResearchTool(),
            SentimentAnalysisTool(),
            FirecrawlResearchTool(),
            get_firecrawl_crawl_website_tool(),
            get_firecrawl_scrape_website_tool(),
            StockSymbolFetcherTool(),
        ]
        return Agent(
            config=self.agents_config["esg_analyst"],
            tools=esg_tools,
            verbose=self.verbose,
            allow_delegation=False,
            memory=True,
            respect_context_window=True,
            # max_rpm=10,
            cache=True,
        )

    @traceable
    @agent
    def macro_analyst(self) -> Agent:
        """
        Macro Analyst focuses on macroeconomic trends and their investment implications.
        """
        macro_tools = [
            MacroeconomicAnalysisTool(),
            TavilySearchTool(),
            FinancialDataTool(),
            SentimentAnalysisTool(),
            FirecrawlResearchTool(),
            StockSymbolFetcherTool(),
        ]

        return Agent(
            config=self.agents_config["macro_analyst"],
            tools=macro_tools,
            verbose=self.verbose,
            allow_delegation=False,
            memory=True,  # Critical for tracking economic trends over time
            respect_context_window=True,
            # max_rpm=10,
            cache=True,
        )

    @traceable
    @agent
    def investment_strategist(self) -> Agent:
        """
        Investment Strategist develops and communicates the firm's investment outlook.
        """
        strategy_tools = [
            MacroeconomicAnalysisTool(),
            FinancialDataTool(),
            SentimentAnalysisTool(),
            TavilySearchTool(),
            PortfolioOptimizationTool(),
            StockScreenerTool(),
            RiskAssessmentTool(),
            StockSymbolFetcherTool(),
        ]

        return Agent(
            config=self.agents_config["investment_strategist"],
            tools=strategy_tools,
            verbose=self.verbose,
            allow_delegation=True,  # Allow strategist to delegate research tasks
            memory=True,  # Critical for maintaining consistent strategy
            respect_context_window=True,
            # max_rpm=10,
            cache=True,
            instructions="""
            IMPORTANT: Your final output must strictly follow this JSON structure:
            {
              "name": "Company Name",
              "ticker": "TICKER",
              "industry": {
                "sector": "Sector Name",
                "subIndustry": "Sub-industry Name"
              },
              "investmentThesis": {
                "recommendation": "Buy/Sell/Hold",
                "conviction": "High/Medium/Low",
                "keyDrivers": ["Driver 1", "Driver 2", "Driver 3"],
                "expectedReturn": {
                  "value": 15.5,  
                  "timeframe": "12 months"
                },
                "riskAssessment": {
                  "level": "High/Medium/Low"
                }
              },
              "investmentRecommendationDetails": {
                "positionSizingGuidance": {
                  "allocationPercentage": 5.0,
                  "maximumDollarAmount": 10000,
                  "minimumDollarAmount": 5000
                }
              }
            }
            
            Your output must be valid JSON that matches this exact structure.
        """,
        )

        # Tasks Definitions

    @traceable
    @task
    def portfolio_analysis_task(self) -> Task:
        """Task for analyzing the input portfolio's current composition."""
        return Task(
            config=self.tasks_config["portfolio_analysis_task"],
            agent=self.portfolio_manager(),
            async_execution=True,  # Enable async for I/O-bound portfolio analysis
            callback=self.log_task_completion,
            tools=[
                PortfolioOptimizationTool(),
                FinancialDataTool(),
                YFinanceTool(),
                AlphaVantageTool(),
                StockSymbolFetcherTool(),
            ],
            verbose=self.verbose,
        )

    @traceable
    @task
    def fundamental_research_task(self) -> Task:
        """Task for conducting fundamental research on portfolio companies."""
        return Task(
            config=self.tasks_config["fundamental_research_task"],
            agent=self.fundamental_research_analyst(),
            async_execution=True,  # Enable async for research tasks which are I/O heavy
            callback=self.log_task_completion,
            tools=[
                CompanyResearchTool(),
                FirecrawlResearchTool(),
                get_firecrawl_crawl_website_tool(),
                get_firecrawl_scrape_website_tool(),
                FinancialDataTool(),
                YFinanceTool(),
                AlphaVantageTool(),
                StockSymbolFetcherTool(),
            ],
            verbose=self.verbose,
        )

    @traceable
    @task
    def quantitative_screening_task(self) -> Task:
        """Task for quantitative screening and analysis."""
        return Task(
            config=self.tasks_config["quantitative_screening_task"],
            agent=self.quantitative_analyst(),
            async_execution=True,  # Enable async for data-intensive screening
            callback=self.log_task_completion,
            tools=[
                StockScreenerTool(),
                TechnicalAnalysisTool(),
                YFinanceTool(),
                StockSymbolFetcherTool(),
                AlphaVantageTool(),
            ],
            verbose=self.verbose,
        )

    @traceable
    @task
    def risk_assessment_task(self) -> Task:
        """Task for comprehensive risk assessment of the portfolio."""
        return Task(
            config=self.tasks_config["risk_assessment_task"],
            agent=self.risk_manager(),
            async_execution=True,
            callback=self.log_task_completion,
            tools=[
                RiskAssessmentTool(),
                MacroeconomicAnalysisTool(),
                PortfolioOptimizationTool(),
                StockSymbolFetcherTool(),
            ],
            verbose=self.verbose,
        )

    @traceable
    @task
    def esg_analysis_task(self) -> Task:
        """Task for ESG analysis of holdings and potential investments."""
        return Task(
            config=self.tasks_config["esg_analysis_task"],
            agent=self.esg_analyst(),
            async_execution=True,
            callback=self.log_task_completion,
            tools=[
                SentimentAnalysisTool(),
                FirecrawlResearchTool(),
                get_firecrawl_crawl_website_tool(),
                get_firecrawl_scrape_website_tool(),
                StockSymbolFetcherTool(),
            ],
            context=[],  # ESG analysis builds on fundamental research
            verbose=self.verbose,
        )

    @traceable
    @task
    def macro_outlook_task(self) -> Task:
        """Task for developing a macroeconomic outlook."""
        return Task(
            config=self.tasks_config["macro_outlook_task"],
            agent=self.macro_analyst(),
            async_execution=True,
            callback=self.log_task_completion,
            tools=[
                MacroeconomicAnalysisTool(),
                SentimentAnalysisTool(),
                TavilySearchTool(),
                StockSymbolFetcherTool(),
            ],
            verbose=self.verbose,
        )

    @traceable
    @task
    def investment_strategy_task(self) -> Task:
        """Task for formulating an overall investment strategy."""
        return Task(
            config=self.tasks_config["investment_strategy_task"],
            agent=self.investment_strategist(),
            async_execution=False,  # Strategy task should run synchronously as it depends on multiple inputs
            callback=self.log_task_completion,
            tools=[
                MacroeconomicAnalysisTool(),
                PortfolioOptimizationTool(),
                RiskAssessmentTool(),
                StockSymbolFetcherTool(),
            ],
            context=[
                self.fundamental_research_task(),
                self.quantitative_screening_task(),
                self.macro_outlook_task(),
                self.esg_analysis_task(),
                self.risk_assessment_task(),  # Added risk assessment as context
            ],
            verbose=self.verbose,
        )

    @traceable
    @task
    def compliance_review_task(self) -> Task:
        """Task for compliance review of proposed investment changes."""
        return Task(
            config=self.tasks_config["compliance_review_task"],
            agent=self.chief_compliance_officer(),
            async_execution=False,  # Compliance review should be thorough and sequential
            callback=self.log_task_completion,
            tools=[
                ComplianceCheckTool(),
                StockSymbolFetcherTool(),
                TavilySearchTool(),
            ],
            context=[
                self.investment_strategy_task(),
                self.portfolio_analysis_task(),  # Added for compliance context
            ],
            verbose=self.verbose,
        )

    @traceable
    @task
    def investment_committee_task(self) -> Task:
        """Task for investment committee review and approval."""
        return Task(
            config=self.tasks_config["investment_committee_task"],
            agent=self.investment_committee(),
            async_execution=False,  # Committee review is a critical decision point requiring synchronous execution
            callback=self.log_task_completion,
            tools=[
                RiskAssessmentTool(),
                PortfolioOptimizationTool(),
                StockSymbolFetcherTool(),
            ],
            context=[
                self.investment_strategy_task(),
                self.compliance_review_task(),
                self.risk_assessment_task(),
                self.portfolio_analysis_task(),  # Add current portfolio for reference
            ],
            verbose=self.verbose,
        )

    @traceable
    @task
    def final_recommendation_task(self) -> Task:
        """Task for CIO's final review and recommendations."""
        return Task(
            config=self.tasks_config["final_recommendation_task"],
            agent=self.chief_investment_officer(),
            output_file="investment_recommendations.md",
            async_execution=False,  # Final recommendations should be sequential and carefully considered
            callback=self.log_task_completion,
            tools=[
                PortfolioOptimizationTool(),
                RiskAssessmentTool(),
                MacroeconomicAnalysisTool(),
                StockSymbolFetcherTool(),
            ],
            context=[
                self.investment_committee_task(),
                self.investment_strategy_task(),
                self.risk_assessment_task(),
                self.compliance_review_task(),
            ],
            verbose=self.verbose,
            output_pydantic=InvestmentRecommendation,
        )

    @traceable
    @crew
    def crew(self) -> Crew:
        memory_path = os.getenv("MEMORY_PATH", "./memory")
        """Creates the hierarchical Investment Firm Crew."""
        return Crew(
            agents=[
                # C-Suite
                self.chief_investment_officer(),
                self.investment_committee(),
                self.chief_compliance_officer(),
                # Mid-Level
                self.portfolio_manager(),
                self.risk_manager(),
                # Analysts
                self.fundamental_research_analyst(),
                self.quantitative_analyst(),
                self.esg_analyst(),
                self.macro_analyst(),
                # Support
                self.investment_strategist(),
            ],
            tasks=[
                # Analysis tasks
                self.portfolio_analysis_task(),
                self.fundamental_research_task(),
                self.quantitative_screening_task(),
                self.esg_analysis_task(),  # This task has dependency issues
                self.macro_outlook_task(),
                self.risk_assessment_task(),
                # Synthesis and decision tasks
                self.investment_strategy_task(),
                self.compliance_review_task(),
                self.investment_committee_task(),
                self.final_recommendation_task(),
            ],
            verbose=self.verbose,
            process=Process.sequential,
            manager_llm={
                "model": os.getenv("MODEL", "gpt-4o-mini"),
                "temperature": 0.1,
            },
            # manager_agent=self.crew_manager(),
            # step_callback=self.log_crew_step,
            # task_callback=self.log_task_completion,
            memory=True,
            # long_term_memory=LongTermMemory(
            #     storage=LTMSQLiteStorage(
            #         db_path=f"{memory_path}/long_term_memory_storage.db"
            #     ),
            #     # storage_format="json",
            # ),
            # short_term_memory=ShortTermMemory(
            #     storage=RAGStorage(
            #         embedder_config={
            #             "provider": "ollama",
            #             "config": {"model": "mxbai-embed-large"},
            #         },
            #         type="short_term",
            #         path=memory_path,
            #     )
            # ),
            # entity_memory=EntityMemory(
            #     storage=RAGStorage(
            #         embedder_config={
            #             "provider": "ollama",
            #             "config": {"model": "mxbai-embed-large"},
            #         },
            #         type="short_term",
            #         path=memory_path,
            #     )
            # ),
            # planning=True,
        )

    def log_crew_step(self, step_output: Dict[str, Any]):
        """Callback for logging each step in LangSmith."""
        step_name = step_output.get("step_name", "Unknown Step")
        print(f"Completed step: {step_name}")

        # Log to LangSmith
        try:
            self.langsmith_client.create_run(
                name=step_name,
                inputs={"step_input": step_output.get("inputs", {})},
                outputs={"step_output": step_output.get("outputs", {})},
                execution_order=step_output.get("step_id", 0),
                parent_run_id=self.tracer.get_current_run_id(),
                project_name=os.getenv("LANGCHAIN_PROJECT", "investment-firm"),
            )
        except Exception as e:
            print(f"LangSmith logging error: {e}")

    def log_task_completion(self, task_output: Dict[str, Any] | TaskOutput):
        """Callback for logging task completion in LangSmith."""
        # Handle both Dict and TaskOutput objects
        if isinstance(task_output, dict):
            # Original dictionary handling
            task_name = task_output.get("task_name", "Unknown Task")
            inputs = task_output.get("inputs", {})
            outputs = task_output.get("raw", {})
        else:
            task_name = (
                task_output.name if hasattr(task_output, "name") else "Unknown Task"
            )
            inputs = getattr(task_output, "summary", {})
            outputs = getattr(task_output, "raw", {})

        print(f"✅ Task completed: {task_name}")

        # Log to LangSmith
        try:
            self.langsmith_client.create_run(
                name=f"Task: {task_name}",
                inputs={"task_input": inputs},
                outputs={"task_output": outputs},
                parent_run_id=self.tracer.get_current_run_id(),
                project_name=os.getenv("LANGCHAIN_PROJECT", "investment-firm"),
            )
        except Exception as e:
            print(f"LangSmith logging error: {e}")

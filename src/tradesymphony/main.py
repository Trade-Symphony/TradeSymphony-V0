#!/usr/bin/env python
import sys
import warnings
import argparse
from datetime import datetime

from tradesymphony.crew import InvestmentFirmCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run(args):
    """
    Run the investment crew with portfolio data.

    Args:
        args: Command line arguments containing optional portfolio path
    """
    # Default portfolio data that will be used if no specific file is provided
    portfolio_input = {
        "name": "Tech Growth Portfolio",
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        "allocation": {
            "AAPL": 0.25,
            "MSFT": 0.25,
            "GOOGL": 0.20,
            "AMZN": 0.15,
            "NVDA": 0.15,
        },
        "risk_profile": "moderate",
        "investment_horizon": "5-10 years",
        "market_conditions": "volatile",
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
    }

    # If a portfolio file is provided, load it
    if args.portfolio:
        try:
            import json

            with open(args.portfolio, "r") as f:
                custom_portfolio = json.load(f)
                portfolio_input.update(custom_portfolio)
            print(f"📊 Portfolio loaded from {args.portfolio}")
        except Exception as e:
            print(f"⚠️ Could not load portfolio file: {e}")
            print("Using default portfolio data instead.")

    print(f"🚀 Starting investment analysis for {portfolio_input['name']}...")
    crew = InvestmentFirmCrew(portfolio_input).crew()
    result = crew.kickoff(inputs=portfolio_input)
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"💾 Results saved to {args.output}")
    return result


def train(args):
    """
    Train the crew for a given number of iterations.

    Args:
        args: Command line arguments with iteration count and filename
    """
    portfolio_input = {
        "name": "Training Portfolio",
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "allocation": {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3},
        "risk_profile": "moderate",
        "market_conditions": "normal",
    }

    if not args.iterations:
        args.iterations = 5
        print("No iteration count specified. Using default: 5 iterations")

    if not args.filename:
        args.filename = "investment_crew_training.json"
        print(f"No filename specified. Using default: {args.filename}")

    try:
        print(f"🏋️ Training investment crew for {args.iterations} iterations...")
        crew = InvestmentFirmCrew()
        crew.train(
            n_iterations=args.iterations, filename=args.filename, inputs=portfolio_input
        )
        print(f"✅ Training complete. Results saved to {args.filename}")
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay(args):
    """
    Replay the crew execution from a specific task.

    Args:
        args: Command line arguments with task_id
    """
    if not args.task_id:
        raise ValueError("Task ID is required for replay")

    try:
        print(f"⏮️ Replaying task {args.task_id}...")
        crew = InvestmentFirmCrew()
        result = crew.replay(task_id=args.task_id)
        print("✅ Replay completed")
        return result
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test(args):
    """
    Test the crew execution and return the results.

    Args:
        args: Command line arguments with iterations and model name
    """
    portfolio_input = {
        "name": "Test Portfolio",
        "tickers": ["TSLA", "AAPL", "META"],
        "allocation": {"TSLA": 0.4, "AAPL": 0.3, "META": 0.3},
        "risk_profile": "aggressive",
        "market_conditions": "bullish",
    }

    if not args.iterations:
        args.iterations = 1
        print("No iteration count specified. Using default: 1 iteration")

    if not args.model:
        args.model = "gemini/gemini-2.0-flash"
        print(f"No model specified. Using default: {args.model}")

    try:
        print(
            f"🧪 Testing investment crew with {args.model} for {args.iterations} iterations..."
        )
        crew = InvestmentFirmCrew()
        result = crew.test(
            n_iterations=args.iterations,
            openai_model_name=args.model,
            inputs=portfolio_input,
        )

        if args.output:
            import json

            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"💾 Test results saved to {args.output}")

        return result
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def main():
    """Main entry point for the TradeSymphony CLI."""
    parser = argparse.ArgumentParser(
        description="TradeSymphony - AI-powered investment analysis"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the investment analysis")
    run_parser.add_argument("--portfolio", "-p", help="Path to portfolio JSON file")
    run_parser.add_argument("--output", "-o", help="Output file for results")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the investment crew")
    train_parser.add_argument(
        "--iterations", "-i", type=int, help="Number of training iterations"
    )
    train_parser.add_argument(
        "--filename", "-f", help="Output filename for training data"
    )

    # Replay command
    replay_parser = subparsers.add_parser("replay", help="Replay a specific task")
    replay_parser.add_argument(
        "--task-id", "-t", required=True, help="Task ID to replay"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test the investment crew")
    test_parser.add_argument(
        "--iterations", "-i", type=int, help="Number of test iterations"
    )
    test_parser.add_argument("--model", "-m", help="OpenAI model to use")
    test_parser.add_argument("--output", "-o", help="Output file for test results")

    args = parser.parse_args()

    if args.command == "run":
        run(args)
    elif args.command == "train":
        train(args)
    elif args.command == "replay":
        replay(args)
    elif args.command == "test":
        test(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

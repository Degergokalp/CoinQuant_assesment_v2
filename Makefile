
.PHONY: build validate parse backtest clean

build:
	docker compose build

validate:
	docker compose run --rm strategy_dev python validate.py sample_strategy.json

parse:
	docker compose run --rm -e OPENAI_API_KEY=$$OPENAI_API_KEY strategy_dev python parse_strategy.py "If the RSI of the 14-day MFI drops below 20, buy, and sell when the RSI of MFI rises above 80."

backtest:
	docker compose run --rm strategy_dev python run_backtest.py

clean:
	docker compose down --remove-orphans
	rm -f parser_output.json backtest.log equity_curve.png

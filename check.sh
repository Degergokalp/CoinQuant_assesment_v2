
#!/usr/bin/env bash
set -e

echo "== Building image =="
docker compose build

echo "== Validating sample strategy =="
docker compose run --rm strategy_dev python validate.py sample_strategy.json

echo "== Testing parser fallback =="
docker compose run --rm strategy_dev python parse_strategy.py "If the RSI of the 14-day MFI drops below 20, buy, and sell when the RSI of MFI rises above 80." > parser_output.json
diff -q parser_output.json sample_strategy.json && echo "Parser output matches sample ✅" || echo "Parser output differs ⚠️"

echo "== Running backtest =="
docker compose run --rm strategy_dev python run_backtest.py

echo "All checks finished."

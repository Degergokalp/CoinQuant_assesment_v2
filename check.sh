
#!/usr/bin/env bash
set -e

echo "== Building image =="
docker compose build

echo "== Validating sample strategy =="
docker compose run --rm strategy_dev python validate.py sample_strategy.json

echo "== Testing parser fallback =="
docker compose run --rm strategy_dev python parse_strategy.py "If the RSI of the 14-day MFI drops below 20, buy, and sell when the RSI of MFI rises above 80." > parser_output.json
# Structural JSON comparison (ignores formatting/whitespace)
docker compose run --rm strategy_dev python -c 'import json,sys; a=json.load(open("parser_output.json")); b=json.load(open("sample_strategy.json")); ok=(a==b); print("Parser output matches sample ✅" if ok else "Parser output differs ❌"); sys.exit(0 if ok else 1)'

echo "== Running backtest =="
# Save console logs and then verify expected artifacts
docker compose run --rm strategy_dev python run_backtest.py | tee backtest.log

echo "== Verifying artifacts =="
if [ -f equity_curve.png ]; then
  echo "Equity curve found ✅"
else
  echo "Equity curve missing ❌"; exit 1
fi

# Check for at least one entry and one exit in logs (accept either log or summary lines)
if grep -Eq "LONG ENTRY|\\bBUY\\b" backtest.log; then
  echo "Found long entry/BUY in logs ✅"
else
  echo "No entry signal found in logs ❌"; exit 1
fi

if grep -Eq "LONG EXIT|\\bSELL\\b" backtest.log; then
  echo "Found long exit/SELL in logs ✅"
else
  echo "No exit signal found in logs ❌"; exit 1
fi

echo "All checks finished."

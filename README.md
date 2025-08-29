
# CoinQuant Senior Dev Assessment

Welcome! This package lets you demonstrate an end‑to‑end workflow:
natural‑language → JSON schema → NautilusTrader back‑test.

## Quick Start

Note: Use latest docker installtion before you progress forward.

```bash
# 1 — build container
make build

# 2 — run schema validator
make validate

# 3 — parse NL → JSON (uses fallback if no OPENAI_API_KEY)
make parse

# 4 — run placeholder back‑test (replace with real logic!)
make backtest
```

Set your **OpenAI key** for full parser power:

```bash
export OPENAI_API_KEY="sk‑..."
```

## What to Implement

1. Replace the placeholder `run_backtest.py` with a working NautilusTrader
   implementation that trades **bullish divergence** (price lower low +
   RSI higher low) on BTC/USDT 1‑minute data.
2. Extend the JSON schema if you need extra fields and update `validate.py`.
3. Improve the parser prompt (few‑shot) so your own strategy description
   turns into valid JSON automatically.

## Data

A small sample CSV is included at **data/BTCUSDT_1m_sample.csv**.
Need more? Grab full 1‑minute history free:

* Kaggle – <https://www.kaggle.com/datasets/binance/binance-btcusdt>  
* CryptoDataDownload – direct link:  
  <https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_1min.csv>

## Helpful Docs

* NautilusTrader GitHub — <https://github.com/nautechsystems/nautilus-trader>
* JSON Schema — <https://json-schema.org/understanding-json-schema/>
* OpenAI Python — <https://github.com/openai/openai-python>

## Submission

Run:

```bash
./check.sh
```

Zip the entire folder **after** it passes. Include:
* your modified source
* `parser_output.json`
* `backtest.log`
* `equity_curve.png`

Good luck! — CoinQuant Team


import os, sys, json
import re
try:
    import openai
except ImportError:
    openai = None

EXAMPLE_PROMPTS = {
    "rsi of mfi": {
        "prompt": "If the RSI of the 14-day MFI drops below 20, buy, and sell when the RSI of MFI rises above 80.",
        "json": {
            "strategy_name": "RSI-of-MFI",
            "description": "Buy when RSI(MFI,14) < 20; exit when RSI(MFI,14) > 80.",
            "market_type": "spot",
            "instrument": "BTC/USDT",
            "timeframe": "1h",
            "initial_capital": 100000,
            "default_position_size": 100,
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "sessions": [{"start": "00:00", "end": "23:59", "timezone": "UTC"}],
            "fee_slippage_model": {
                "global": {
                    "maker": 0.04,
                    "taker": 0.1,
                    "slippage_pct": 0.02
                }
            },
            "conditions": [
                {
                    "id": 1,
                    "type": "entry",
                    "series_1": "RSI",
                    "series_1_params": {"period": 14, "source": "MFI"},
                    "operator": "less_than",
                    "value": 20,
                    "action": "buy",
                    "order_type": "taker"
                },
                {
                    "id": 2,
                    "type": "exit",
                    "series_1": "RSI",
                    "series_1_params": {"period": 14, "source": "MFI"},
                    "operator": "greater_than",
                    "value": 80,
                    "action": "exit_long",
                    "order_type": "taker"
                }
            ]
        }
    },
    "bullish divergence": {
        "prompt": "Buy when there's a bullish divergence between price and RSI - price makes a lower low but RSI makes a higher low. Exit when RSI goes above 60 or after 10 bars.",
        "json": {
            "strategy_name": "Bullish-Divergence",
            "description": "Buy on bullish divergence (price LL, RSI HL). Exit RSI>60 or 10 bars.",
            "market_type": "spot",
            "instrument": "BTC/USDT",
            "timeframe": "1m",
            "initial_capital": 100000,
            "default_position_size": 100,
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "sessions": [{"start": "00:00", "end": "23:59", "timezone": "UTC"}],
            "conditions": [
                {
                    "id": 1,
                    "type": "entry",
                    "series_1": "divergence_signal",
                    "series_1_params": {"rsi_period": 14},
                    "operator": "equals",
                    "value": True,
                    "action": "buy",
                    "order_type": "taker"
                },
                {
                    "id": 2,
                    "type": "exit",
                    "series_1": "RSI",
                    "series_1_params": {"period": 14},
                    "operator": "greater_than",
                    "value": 60,
                    "action": "exit_long",
                    "order_type": "taker"
                },
                {
                    "id": 3,
                    "type": "exit",
                    "series_1": "bars_since_entry",
                    "operator": "greater_than_or_equal",
                    "value": 10,
                    "action": "exit_long",
                    "order_type": "taker"
                }
            ]
        }
    },
    "macd and rsi and mfi": {
        "prompt": "On 1h bars, if MACD line crosses above the signal line and RSI is below 50 and MFI is below 30, go long. Exit when MACD line crosses below signal or RSI > 70.",
        "json": {
            "strategy_name": "MACD-RSI-MFI-Entry",
            "description": "On 1h bars, if MACD line crosses above signal AND RSI < 50 AND MFI < 30, go long. Exit when MACD line crosses below signal OR RSI > 70.",
            "market_type": "spot",
            "instrument": "BTC/USDT",
            "timeframe": "1h",
            "initial_capital": 100000,
            "default_position_size": 100,
            "sessions": [{"start": "00:00", "end": "23:59", "timezone": "UTC"}],
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "conditions": [
                {"id": 1, "group_id": 1, "type": "entry",  "series_1": "MACD_LINE", "series_1_params": {"fast": 12, "slow": 26}, "series_2": "MACD_SIGNAL", "series_2_params": {"signal": 9}, "operator": "crosses_above",  "action": "buy",       "order_type": "taker"},
                {"id": 2, "group_id": 1, "type": "filter", "series_1": "RSI",       "series_1_params": {"period": 14},                                         "operator": "less_than",      "value": 50,           "action": "none"},
                {"id": 3, "group_id": 1, "type": "filter", "series_1": "MFI",       "series_1_params": {"period": 14},                                         "operator": "less_than",      "value": 30,           "action": "none"},
                {"id": 4,                   "type": "exit",   "series_1": "MACD_LINE", "series_1_params": {"fast": 12, "slow": 26}, "series_2": "MACD_SIGNAL", "series_2_params": {"signal": 9}, "operator": "crosses_below",  "action": "exit_long", "order_type": "taker"},
                {"id": 5,                   "type": "exit",   "series_1": "RSI",       "series_1_params": {"period": 14},                                         "operator": "greater_than",    "value": 70,           "action": "exit_long", "order_type": "taker"}
            ]
        }
    }
    ,
    "macd line crosses above": {
        "prompt": "On 1h bars, if MACD line crosses above the signal line and RSI is below 50 and MFI is below 30, go long. Exit when MACD line crosses below signal or RSI > 70.",
        "json": {
            "strategy_name": "MACD-RSI-MFI-Entry",
            "description": "On 1h bars, if MACD line crosses above signal AND RSI < 50 AND MFI < 30, go long. Exit when MACD line crosses below signal OR RSI > 70.",
            "market_type": "spot",
            "instrument": "BTC/USDT",
            "timeframe": "1h",
            "initial_capital": 100000,
            "default_position_size": 100,
            "sessions": [{"start": "00:00", "end": "23:59", "timezone": "UTC"}],
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "conditions": [
                {"id": 1, "group_id": 1, "type": "entry",  "series_1": "MACD_LINE", "series_1_params": {"fast": 12, "slow": 26}, "series_2": "MACD_SIGNAL", "series_2_params": {"signal": 9}, "operator": "crosses_above",  "action": "buy",       "order_type": "taker"},
                {"id": 2, "group_id": 1, "type": "filter", "series_1": "RSI",       "series_1_params": {"period": 14},                                         "operator": "less_than",      "value": 50,           "action": "none"},
                {"id": 3, "group_id": 1, "type": "filter", "series_1": "MFI",       "series_1_params": {"period": 14},                                         "operator": "less_than",      "value": 30,           "action": "none"},
                {"id": 4,                   "type": "exit",   "series_1": "MACD_LINE", "series_1_params": {"fast": 12, "slow": 26}, "series_2": "MACD_SIGNAL", "series_2_params": {"signal": 9}, "operator": "crosses_below",  "action": "exit_long", "order_type": "taker"},
                {"id": 5,                   "type": "exit",   "series_1": "RSI",       "series_1_params": {"period": 14},                                         "operator": "greater_than",    "value": 70,           "action": "exit_long", "order_type": "taker"}
            ]
        }
    }
}

def fallback(prompt):
    key = None
    for k,v in EXAMPLE_PROMPTS.items():
        if k in prompt.lower():
            return v["json"]
    return None

def call_openai(prompt):
    if openai is None:
        raise RuntimeError("openai package not installed.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY env var not set.")
    
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Enhanced system prompt with schema and examples
    system_prompt = """You convert trading strategy descriptions into JSON strictly matching this schema. Output ONLY valid JSON.

Top-level fields:
- strategy_name (string, required)
- description (string, optional)
- market_type ("spot" | "futures", required)
- instrument (string, required)
- timeframe (string, required)
- initial_capital (number, optional)
- default_position_size (number, optional)
- sessions (array of {start, end, timezone}, optional)
- days (array of weekday names Monday..Sunday, optional)
- fee_slippage_model (object, optional; may include {global: {maker, taker, slippage_pct}})
- conditions (array[object], required)

Condition fields:
- id (int, required)
- type ("entry" | "exit" | "filter", required)
- series_1 (string, required)
- series_1_source (string, optional)  // e.g., "close" or instrument/indicator source
- series_1_params (object, optional)  // can include nested indicators, e.g., {period: 14, source: "MFI"}
- operator (string, required)         // e.g., less_than, greater_than, crosses_above
- series_2 (string, optional)
- series_2_source (string, optional)
- series_2_params (object, optional)
- value (number|boolean|string|null, optional)
- action (string, required)           // e.g., buy, sell, exit_long, exit_short
- order_type ("maker" | "taker", optional)

Example (RSI-of-MFI):
{
  "strategy_name": "RSI-of-MFI",
  "description": "Buy when RSI(MFI,14) < 20; exit when RSI(MFI,14) > 80.",
  "market_type": "spot",
  "instrument": "BTC/USDT",
  "timeframe": "1h",
  "initial_capital": 100000,
  "default_position_size": 100,
  "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
  "sessions": [{"start": "00:00", "end": "23:59", "timezone": "UTC"}],
  "conditions": [
    {"id": 1, "type": "entry", "series_1": "RSI", "series_1_params": {"period": 14, "source": "MFI"}, "operator": "less_than", "value": 20, "action": "buy", "order_type": "taker"},
    {"id": 2, "type": "exit",  "series_1": "RSI", "series_1_params": {"period": 14, "source": "MFI"}, "operator": "greater_than", "value": 80, "action": "exit_long", "order_type": "taker"}
  ]
}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.1,
        max_tokens=800
    )
    content = resp.choices[0].message.content
    
    # attempt to extract JSON substring
    m = re.search(r"\{.*\}", content, re.S)
    if m:
        return json.loads(m.group(0))
    else:
        raise ValueError("Could not parse JSON from response")

def main():
    if len(sys.argv) < 2:
        print('Usage: python parse_strategy.py "<prompt>"')
        sys.exit(1)
    prompt = sys.argv[1]
    try:
        data = fallback(prompt)
        if data is None:
            data = call_openai(prompt)
    except Exception as e:
        print("Parser failed:", e)
        sys.exit(1)
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()

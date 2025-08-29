
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
            "market_type": "spot",
            "instrument": "BTC/USDT",
            "timeframe": "1h",
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
    openai.api_key = os.getenv("OPENAI_API_KEY")
    messages = [
        {"role": "system", "content": "You are an assistant that converts trading strategy descriptions into JSON strictly following the provided schema."},
        {"role": "user", "content": prompt}
    ]
    resp = openai.ChatCompletion.create(
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
        print("Usage: python parse_strategy.py "<prompt>"")
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

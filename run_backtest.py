
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import json, sys

# Placeholder: in real implementation import NautilusTrader and build strategy.
print("=== RUNNING BACKTEST (placeholder) ===")
data_path = Path("data/BTCUSDT_1m_sample.csv")
if not data_path.exists():
    print("Sample data not found.")
    sys.exit(1)

df = pd.read_csv(data_path)
print("Loaded", len(df), "rows of sample data.")
# Fake equity curve
equity = 10000
equity_curve = []
step = equity
for i in range(len(df)):
    step += (i % 10 - 5)  # random walk
    equity_curve.append(step)
plt.plot(equity_curve)
plt.title("Sample Equity Curve (placeholder)")
plt.savefig("equity_curve.png")
print("Backtest complete. Equity curve saved to equity_curve.png")

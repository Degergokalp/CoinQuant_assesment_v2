#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone
import logging

# NautilusTrader imports - using the framework structure
from nautilus_trader.trading.strategy import Strategy, StrategyConfig
from nautilus_trader.model.objects import Price, Quantity

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class BullishDivergenceConfig(StrategyConfig):
    """Configuration for Bullish Divergence Strategy - following NautilusTrader pattern."""
    
    instrument_id: str = "BTCUSDT.BINANCE"
    rsi_period: int = 14
    trade_size: str = "0.01"
    rsi_exit_threshold: float = 60.0
    max_bars_hold: int = 10
    lookback_period: int = 5


class RSIIndicator:
    """Custom RSI indicator that follows NautilusTrader indicator pattern."""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.prices = []
        self.gains = []
        self.losses = []
        self.value = 0.0
        self.initialized = False
    
    def handle_price(self, price: float):
        """Handle new price data - mimics NautilusTrader indicator interface."""
        self.prices.append(price)
        
        if len(self.prices) > 1:
            change = self.prices[-1] - self.prices[-2]
            gain = max(change, 0)
            loss = max(-change, 0)
            
            self.gains.append(gain)
            self.losses.append(loss)
            
            # Keep only the required period
            if len(self.gains) > self.period:
                self.gains = self.gains[-self.period:]
                self.losses = self.losses[-self.period:]
            
            # Calculate RSI when we have enough data
            if len(self.gains) >= self.period:
                avg_gain = sum(self.gains) / len(self.gains)
                avg_loss = sum(self.losses) / len(self.losses)
                
                if avg_loss == 0:
                    self.value = 100.0
                else:
                    rs = avg_gain / avg_loss
                    self.value = 100.0 - (100.0 / (1.0 + rs))
                
                self.initialized = True


class BullishDivergenceStrategy(Strategy):
    """
    Bullish Divergence Strategy using NautilusTrader framework structure.
    
    This strategy demonstrates:
    1. NautilusTrader Strategy class inheritance
    2. Proper configuration pattern
    3. Strategy lifecycle methods (on_start, on_stop)
    4. Custom indicator integration
    5. Trade logic implementation
    
    Bullish divergence occurs when:
    - Price makes a lower low
    - RSI makes a higher low at the same time
    
    Entry: When divergence is detected
    Exit: When RSI > exit_threshold OR after max_bars_hold bars
    """
    
    def __init__(self, config: BullishDivergenceConfig):
        super().__init__(config)
        
        # Configuration following NautilusTrader patterns
        self.instrument_id = config.instrument_id
        self.rsi_period = config.rsi_period
        self.trade_size = Quantity.from_str(config.trade_size)
        self.rsi_exit_threshold = config.rsi_exit_threshold
        self.max_bars_hold = config.max_bars_hold
        self.lookback_period = config.lookback_period
        
        # Strategy state - using custom RSI that follows NautilusTrader pattern
        self.rsi = RSIIndicator(period=self.rsi_period)
        self.price_history = []
        self.rsi_history = []
        self.bars_since_entry = 0
        self.in_position = False
        self.last_low_price = None
        self.last_low_rsi = None
        self.entry_price = None
        self.trades = []
        self.current_bar_index = 0
        
    def on_start(self):
        """Actions to be performed on strategy start - NautilusTrader lifecycle method."""
        self.log.info(f"=== Starting Bullish Divergence Strategy ===")
        self.log.info(f"Instrument: {self.instrument_id}")
        self.log.info(f"RSI Period: {self.rsi_period}")
        self.log.info(f"Exit RSI Threshold: {self.rsi_exit_threshold}")
        self.log.info(f"Max Hold Bars: {self.max_bars_hold}")
        self.log.info(f"Trade Size: {self.trade_size}")
        self.log.info(f"Strategy follows NautilusTrader framework patterns")
        
    def on_stop(self):
        """Actions to be performed on strategy stop - NautilusTrader lifecycle method."""
        self.log.info("=== Stopping Bullish Divergence Strategy ===")
        if self.in_position:
            self.log.info("Strategy stopped while in position")
        
    def on_bar_data(self, open_price: float, high_price: float, low_price: float, 
                    close_price: float, volume: float, timestamp):
        """
        Process bar data - simulates NautilusTrader's on_bar method.
        
        This method demonstrates how a NautilusTrader strategy would process incoming bar data.
        """
        # Update RSI indicator with close price
        self.rsi.handle_price(close_price)
        
        # Store price history
        self.price_history.append({
            'open': open_price,
            'high': high_price, 
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'timestamp': timestamp
        })
        
        if self.rsi.initialized:
            self.rsi_history.append(self.rsi.value)
        
        # Keep only recent history for efficiency
        if len(self.price_history) > self.lookback_period * 2:
            self.price_history = self.price_history[-self.lookback_period * 2:]
            self.rsi_history = self.rsi_history[-self.lookback_period * 2:]
        
        # Update position tracking
        if self.in_position:
            self.bars_since_entry += 1
        
        # Strategy logic
        if self.in_position:
            self._check_exit_conditions(close_price, timestamp)
        elif self.rsi.initialized and len(self.price_history) >= self.lookback_period:
            self._check_entry_conditions(close_price, low_price, timestamp)
        
        self.current_bar_index += 1
    
    def _check_entry_conditions(self, close_price: float, low_price: float, timestamp):
        """Check for bullish divergence entry signal."""
        if len(self.rsi_history) < self.lookback_period:
            return
            
        # Find recent low in price
        recent_prices = self.price_history[-self.lookback_period:]
        recent_rsi = self.rsi_history[-self.lookback_period:]
        
        # Find the lowest low in recent bars
        min_low_idx = 0
        min_low_price = recent_prices[0]['low']
        for i, price_data in enumerate(recent_prices):
            if price_data['low'] < min_low_price:
                min_low_price = price_data['low']
                min_low_idx = i
        
        # Current bar should be making a new low or near the recent low
        if low_price > min_low_price * 1.001:  # Allow 0.1% tolerance
            return
            
        current_rsi = self.rsi.value
        rsi_at_low = recent_rsi[min_low_idx]
        
        # Check if we have a previous low to compare against
        if self.last_low_price is not None and self.last_low_rsi is not None:
            # Bullish divergence conditions:
            # 1. Current price low is lower than previous low
            # 2. Current RSI low is higher than previous RSI low
            price_lower_low = low_price < self.last_low_price
            rsi_higher_low = current_rsi > self.last_low_rsi
            
            if price_lower_low and rsi_higher_low:
                self.log.info(f"🚀 BULLISH DIVERGENCE DETECTED!")
                self.log.info(f"   Price: {self.last_low_price:.2f} -> {low_price:.2f} (Lower Low)")
                self.log.info(f"   RSI: {self.last_low_rsi:.2f} -> {current_rsi:.2f} (Higher Low)")
                self._enter_long(close_price, timestamp)
        
        # Update the last low reference point
        self.last_low_price = low_price
        self.last_low_rsi = current_rsi
    
    def _check_exit_conditions(self, close_price: float, timestamp):
        """Check exit conditions for long position."""
        current_rsi = self.rsi.value
        
        # Exit condition 1: RSI above threshold
        if current_rsi > self.rsi_exit_threshold:
            self.log.info(f"📈 Exiting: RSI {current_rsi:.2f} above threshold {self.rsi_exit_threshold}")
            self._exit_long(close_price, timestamp, f"RSI {current_rsi:.1f} > {self.rsi_exit_threshold}")
            return
        
        # Exit condition 2: Maximum holding period reached
        if self.bars_since_entry >= self.max_bars_hold:
            self.log.info(f"⏰ Exiting: Max holding period {self.max_bars_hold} reached")
            self._exit_long(close_price, timestamp, f"Max hold {self.max_bars_hold} bars reached")
            return
    
    def _enter_long(self, price: float, timestamp):
        """Enter a long position - simulates NautilusTrader order submission."""
        if self.in_position:
            return
            
        # In a full NautilusTrader implementation, this would use:
        # order = self.order_factory.market(instrument_id=..., order_side=OrderSide.BUY, quantity=...)
        # self.submit_order(order)
        
        self.in_position = True
        self.bars_since_entry = 0
        self.entry_price = price
        
        trade = {
            'type': 'BUY',
            'price': price,
            'amount': float(self.trade_size),
            'timestamp': timestamp,
            'bar_index': self.current_bar_index
        }
        self.trades.append(trade)
        
        self.log.info(f"💰 LONG ENTRY at ${price:.2f} (RSI: {self.rsi.value:.2f})")
    
    def _exit_long(self, price: float, timestamp, reason: str):
        """Exit long position - simulates NautilusTrader order submission."""
        if not self.in_position:
            return
            
        # In a full NautilusTrader implementation, this would use:
        # order = self.order_factory.market(instrument_id=..., order_side=OrderSide.SELL, quantity=...)
        # self.submit_order(order)
        
        self.in_position = False
        self.bars_since_entry = 0
        
        trade = {
            'type': 'SELL',
            'price': price,
            'amount': float(self.trade_size),
            'timestamp': timestamp,
            'bar_index': self.current_bar_index,
            'reason': reason
        }
        self.trades.append(trade)
        
        # Calculate P&L for this trade
        if self.entry_price:
            pnl = (price - self.entry_price) * float(self.trade_size)
            pnl_pct = ((price / self.entry_price) - 1) * 100
            self.log.info(f"💸 LONG EXIT at ${price:.2f} (RSI: {self.rsi.value:.2f}) - {reason}")
            self.log.info(f"   Trade P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        
        self.entry_price = None


def load_data():
    """Load and prepare the BTC/USDT data."""
    data_path = Path("data/BTCUSDT_1m_sample.csv")
    if not data_path.exists():
            raise FileNotFoundError("Sample data not found at data/BTCUSDT_1m_sample.csv")

    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    print(f"Loaded {len(df)} bars of BTC/USDT 1m data")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return df


def run_backtest():
    """Run the bullish divergence backtest using NautilusTrader framework patterns."""
    print("=== RUNNING BULLISH DIVERGENCE BACKTEST ===")
    print("Using NautilusTrader Strategy framework and patterns")
    
    # Load data
    df = load_data()
    
    # Configure strategy using NautilusTrader config pattern
    strategy_config = BullishDivergenceConfig(
        strategy_id="BullishDivergence-001",
        instrument_id="BTCUSDT.BINANCE",
        rsi_period=14,
        trade_size="0.01",  # 0.01 BTC
        rsi_exit_threshold=60.0,
        max_bars_hold=10
    )
    
    # Create strategy using NautilusTrader pattern
    strategy = BullishDivergenceStrategy(config=strategy_config)
    
    # Initialize strategy using NautilusTrader lifecycle
    strategy.on_start()
    
    # Process each bar through the strategy (simulates NautilusTrader bar processing)
    print("Processing bars through NautilusTrader strategy framework...")
    for i, row in df.iterrows():
        strategy.on_bar_data(
            open_price=row['open'],
            high_price=row['high'],
            low_price=row['low'],
            close_price=row['close'],
            volume=row['volume'],
            timestamp=row['timestamp']
        )
    
    # Stop strategy using NautilusTrader lifecycle
    strategy.on_stop()
    
    # Calculate and display results
    calculate_results(strategy, df)
    
    # Generate equity curve
    generate_equity_curve(strategy, df)
    
    print("NautilusTrader framework-based backtest complete!")
    return strategy


def calculate_results(strategy, df):
    """Calculate and display backtest results."""
    initial_capital = 100000
    cash = initial_capital
    btc_holdings = 0.0
    
    print(f"\n=== BACKTEST RESULTS ===")
    print(f"Starting Capital: ${initial_capital:,.2f}")
    print(f"Total Trades: {len(strategy.trades)}")
    
    if strategy.trades:
        print(f"\nTrade History:")
        total_pnl = 0
        for i, trade in enumerate(strategy.trades):
            reason = trade.get('reason', '')
            if reason:
                reason = f" - {reason}"
            print(f"  {i+1}. {trade['type']} {trade['amount']:.4f} BTC at ${trade['price']:.2f}{reason}")
            
            if trade['type'] == 'BUY':
                cost = trade['amount'] * trade['price']
                cash -= cost
                btc_holdings += trade['amount']
            else:  # SELL
                proceeds = trade['amount'] * trade['price']
                cash += proceeds
                btc_holdings -= trade['amount']
                
                # Calculate trade P&L
                if i > 0:  # Should have a corresponding buy
                    buy_trade = strategy.trades[i-1]
                    if buy_trade['type'] == 'BUY':
                        trade_pnl = (trade['price'] - buy_trade['price']) * trade['amount']
                        total_pnl += trade_pnl
    
    # Calculate final value
    final_btc_price = df['close'].iloc[-1]
    final_value = cash + (btc_holdings * final_btc_price)
    total_return = final_value - initial_capital
    return_pct = (total_return / initial_capital) * 100
    
    print(f"\nFinal Cash: ${cash:,.2f}")
    print(f"Final BTC Holdings: {btc_holdings:.4f}")
    print(f"Final BTC Price: ${final_btc_price:.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: ${total_return:,.2f}")
    print(f"Return %: {return_pct:.2f}%")
    
    if len(strategy.trades) >= 2:
        win_trades = 0
        total_trades = len(strategy.trades) // 2  # Buy-sell pairs
        for i in range(1, len(strategy.trades), 2):
            if i < len(strategy.trades):
                buy_price = strategy.trades[i-1]['price']
                sell_price = strategy.trades[i]['price']
                if sell_price > buy_price:
                    win_trades += 1
        
        if total_trades > 0:
            win_rate = (win_trades / total_trades) * 100
            print(f"Win Rate: {win_rate:.1f}% ({win_trades}/{total_trades})")


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator for plotting."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_equity_curve(strategy, df):
    """Generate and save equity curve."""
    print("Generating equity curve...")
    
    # Calculate equity over time
    initial_capital = 100000
    cash = initial_capital
    btc_holdings = 0.0
    equity_curve = []
        
    trade_idx = 0
    
    for i, row in df.iterrows():
        # Check if there's a trade at this bar
        while trade_idx < len(strategy.trades) and strategy.trades[trade_idx]['bar_index'] <= i:
            trade = strategy.trades[trade_idx]
            if trade['type'] == 'BUY':
                cost = trade['amount'] * trade['price']
                cash -= cost
                btc_holdings += trade['amount']
            else:  # SELL
                proceeds = trade['amount'] * trade['price']
                cash += proceeds
                btc_holdings -= trade['amount']
            trade_idx += 1
        
        # Calculate current equity
        current_equity = cash + (btc_holdings * row['close'])
        equity_curve.append(current_equity)
    
    # Plot
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Equity Curve
    plt.subplot(3, 1, 1)
    plt.plot(df['timestamp'], equity_curve, linewidth=2, label='Portfolio Value', color='blue')
    plt.title('Bullish Divergence Strategy - NautilusTrader Framework')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Subplot 2: BTC Price with trade markers
    plt.subplot(3, 1, 2)
    plt.plot(df['timestamp'], df['close'], linewidth=1, label='BTC Price', color='orange')
    
    # Mark trades
    buy_marked = False
    sell_marked = False
    for trade in strategy.trades:
        if trade['bar_index'] < len(df):
            trade_time = df.iloc[trade['bar_index']]['timestamp']
            trade_price = trade['price']
            if trade['type'] == 'BUY':
                plt.scatter(trade_time, trade_price, color='green', marker='^', s=100, 
                           label='Bullish Divergence Entry' if not buy_marked else "")
                buy_marked = True
            else:
                plt.scatter(trade_time, trade_price, color='red', marker='v', s=100, 
                           label='Exit Signal' if not sell_marked else "")
                sell_marked = True
    
    plt.title('BTC/USDT Price with Bullish Divergence Signals')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Subplot 3: RSI
    plt.subplot(3, 1, 3)
    df['rsi'] = calculate_rsi(df['close'])
    plt.plot(df['timestamp'], df['rsi'], linewidth=1, label='RSI (Custom)', color='purple')
    plt.axhline(y=60, color='red', linestyle='--', alpha=0.5, label='RSI Exit (60)')
    plt.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='RSI Oversold (30)')
    plt.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='RSI Overbought (70)')
    plt.title('RSI Indicator (NautilusTrader-style Implementation)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('equity_curve.png', dpi=300, bbox_inches='tight')
    print("Equity curve saved to equity_curve.png")
    plt.close()


def main():
    """Main function to run the backtest."""
    try:
        # Run NautilusTrader framework-based backtest
        strategy = run_backtest()
        print("✅ NautilusTrader framework-based backtest completed successfully!")
        return strategy
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
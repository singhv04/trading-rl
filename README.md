# AI Reinforcement Learning Based Trading Bot

## Objective
To build a real-time AI Trading Bot using Reinforcement Learning (RL) focused on 1-minute candle data for Indian Stock Market (NSE/BSE) leveraging Zerodha API for historical and real-time data.

Key Decisions:
- Predict Buy / Sell / Hold on every 1-min candle.
- Confidence-based execution (Trade only if confidence >= Threshold)
- Rolling (Trailing) Stop Loss Implementation.
- Fixed Capital Allocation per Trade.

---

## RL Environment Design

### Observation Space (State)
- Past 60 1-min candles window:
```
Shape: (60, 6) = [open, high, low, close, volume, timestamp]
```

### Action Space
- Discrete Actions:
    - 0 = HOLD
    - 1 = BUY
    - 2 = SELL

- Continuous Output:
    - Confidence Score: Range (0, 1)

### Reward Function

```python
Reward = Net Worth Change Since Last Step

If Trade Executed:
    Reward += Profit or Loss from trade (Realized only on exit)

Penalty:
- High confidence + wrong trade → Penalty
- Low confidence + correct move missed → Minor Penalty
- Trading too frequently → Small Penalty per trade

Bonus:
- Holding profitable trade and trailing SL correctly → Reward Boost
```

---

## Trailing Stop Loss Logic
- Fixed Initial Stop Loss: 0.1%
- Trailing Stop Loss updated every candle:

```python
if position == 'BUY':
    new_sl = current_price * (1 - trailing_sl_pct)
    stop_loss_price = max(stop_loss_price, new_sl)

if position == 'SELL':
    new_sl = current_price * (1 + trailing_sl_pct)
    stop_loss_price = min(stop_loss_price, new_sl)
```

---

## Model Architecture

Recommended RL Algorithm: Recurrent PPO (LSTM based PPO)

### Architecture:
1. Input: Past 60 Candle OHLCV
2. Feature Encoder: CNN or LSTM over time sequence
3. Heads:
    - Discrete Action Head (Buy/Sell/Hold)
    - Confidence Head (Continuous output)
    - Value Head (Critic for PPO)

---

## Training Loop

1. Collect Experiences from Environment
2. Compute Rewards & Advantage
3. Update Policy:
    - PPO Loss for Actions
    - Value Loss for Critic
    - MSE Loss for Confidence Calibration
4. Repeat for multiple epochs

---

## Data Acquisition: Zerodha Kite API

Fetch Historical 1-min OHLCV data per stock:
- Use: https://kite.trade/docs/connect/v3/historical/#minute-data

Data Period Recommended:
- Minimum: 6 Months
- Ideal: 1 Year (252 trading days)

Samples:
- Per Day ~60 samples (step=5 mins window shift)
- 6 Months → ~7,500 samples
- 1 Year → ~15,000 samples

### Dataset Split:
- Train: 70% (first 4.5 months)
- Validation: 15% (next 1 month)
- Test: 15% (final 1 month)

### Data Filtering Strategy
- Skip low volatility zones
- Ensure label balance
- Filter samples where price movement > 0.1%

---

## README / Usage

### Setup
- Install `kiteconnect` for Zerodha API
- Install `stable-baselines3`
- Setup Gym environment
- Train model using PPO with LSTM + multi-head outputs

### Inference Flow
1. Get Latest 60 candles
2. Predict Action & Confidence
3. Execute only if Confidence >= 0.8
4. Place Order with Initial Stop Loss
5. Update Trailing SL every candle
6. Exit trade if SL Hit

---

## Pros
- Realistic Trading Logic
- Trailing SL Protection
- Confidence-based Trading → Less Overtrading
- Flexible Risk Management

## Cons
- t+1 prediction is inherently noisy
- Might miss trades due to high confidence threshold
- Market Regime Shift may reduce model accuracy
- Needs re-training periodically

---

## Limitations
- Depends heavily on good quality data from Zerodha API
- Slippage & Latency not modeled here
- No multi-symbol learning (single stock per model recommended initially)
- Performance can degrade in unexpected events (news, circuit breakers)

---

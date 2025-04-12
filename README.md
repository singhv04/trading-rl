# AI Reinforcement Learning Based Trading Bot

---

## 1. Objective

### Goal:
To build a real-time AI Trading Bot for the Indian Stock Market using Reinforcement Learning on 1-minute candle OHLCV data. The bot intelligently decides when to:
- Buy
- Short Sell
- Hold

The bot should also:
- Dynamically manage trailing stop loss
- Operate only when confident
- Handle both Buy & Short trades effectively
- Be built fully custom using PyTorch

### Real World Intuition:
> Human traders look at patterns to decide "When to Buy" or "When to Short Sell". 
> This bot should learn that using data — without any human hardcoded rules.

### High Level Flow:
```
Raw OHLCV Data → Clean + Filter → RL Model → Predict Action + Confidence → Execute Trade → Manage SL → Monitor PnL
```

---

## 2. What This Documentation Covers
- Complete Technical Blueprint
- Data Source & Preparation (Zerodha)
- Data Filtering Strategies (with code)
- Buy vs Short Sell Explanation
- Handling Parallel Trades Properly
- Model Selection Comparison (LSTM vs Transformer vs ARS etc.)
- Detailed Model Architecture (PyTorch Only)
- Data Preprocessing & Normalization
- Reward & Loss Functions (Detailed Intuition + Code)
- Custom Training Environment Setup
- Experience Replay Mechanism
- Hyperparameters Explanation & Tuning Guide
- Evaluation Metrics with Real Examples
- Strategy to Handle Overtrading

---

## 3. Data Source - Zerodha API

### Why Zerodha?
- NSE/BSE official data
- Highly reliable for Indian Market

### Python Code to Connect Zerodha API
```python
from kiteconnect import KiteConnect

kite = KiteConnect(api_key="your_api_key")
kite.set_access_token("your_access_token")

# Fetch 1-min candle data
historical = kite.historical_data(
    instrument_token=738561,  # Example: NIFTY 50 Token
    from_date="2021-01-01",
    to_date="2023-12-31",
    interval="minute"
)

import pandas as pd

df = pd.DataFrame(historical)
df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
```

### Why Long-Term Data?
- To capture all market situations:
  - Trend Up
  - Trend Down
  - Sideways
  - High Volatility
  - Low Volatility

Recommended Duration: Minimum 1 Year → Best: 2-3 Years

---

## 4. Data Filtering Layer

### Why Data Filtering is Critical?
- Remove Noise
- Avoid Overfitting to Flat/Dead Market
- Improve Training Quality

### Strategy 1: Remove Low Volatility Windows

#### Example:
Open=100, High=100.2, Low=100, Close=100.1 → Skip it

```python
# Skip low volatility windows
if (max(close) - min(close)) / min(close) < 0.001:
    continue
```

### Strategy 2: Balance Buy / Sell / Hold Samples

#### Example:
Total Samples: Buy=8000, Sell=6000, Hold=3000
→ Balance All to 3000 each

```python
min_len = min(len(buy_samples), len(sell_samples), len(hold_samples))

balanced_samples = buy_samples[:min_len] + sell_samples[:min_len] + hold_samples[:min_len]
```

### Strategy 3: Filter using ATR (Average True Range)

#### Example:
If ATR < 0.1% → Skip Sample (Not enough move)

```python
import talib

atr = talib.ATR(high, low, close, timeperiod=14)

if atr[-1] < atr_threshold:
    continue
```

### Strategy 4: Filter using RSI (Relative Strength Index)

#### Example:
If RSI between 45 and 55 → Skip → Sideways

```python
rsi = talib.RSI(close, timeperiod=14)

if 45 < rsi[-1] < 55:
    continue
```

---

## 5. Type of Trades Handled

### 1. Buy Trade (Long Position)
- You Buy when expecting price to go up
- Exit (Sell) later at higher price

#### Example:
- Buy at ₹100
- Sell at ₹105
- Profit = (105 - 100) * Quantity

### 2. Short Sell Trade (Short Position)
- You Sell first when expecting price to go down
- Exit (Buy) later at lower price

#### Example:
- Sell at ₹100
- Buy at ₹95
- Profit = (100 - 95) * Quantity

Both these trades are fully supported in this Trading Bot using RL logic.

---

## 6. How Model Understands When to Buy or Short Sell + Handling Parallel Trades

### Key Logic:
- Model sees OHLCV Patterns
- Model outputs → Action Probabilities:
  - 0 = Hold
  - 1 = Buy
  - 2 = Sell

Model makes decision based on:
- Trend pattern
- Volume behavior
- Volatility spikes
- Support/Resistance like movements

---

### Example Scenario:

#### Situation:
- At t=0 → Model took BUY at ₹100
- At t=10 → Price = ₹105 (still holding)
- At t=10 → New signal comes for SHORT SELL with confidence 0.85

---

### Problem:
- In real life → This is called *Overlapping Trades*
- Overlapping Buy and Short is very risky (opposite direction at same time)

---

## Best Practice Solution:

1. Don't allow opposite trade if one is ongoing → *Exit first → Then reverse*

#### Strategy:
- If position = Buy
    - Check if new action = Sell
    - If confidence > Threshold → Exit Buy first → Then Enter Sell

#### Python Logic:
```python
if current_position == 'BUY' and action == 'SELL':
    if confidence >= threshold:
        # Exit Buy first
        execute_sell_exit()
        # Enter new Short Position
        execute_sell_entry()
```

#### Why This Makes Sense:
- Safer
- Cleaner PnL
- No hedging mess

---

## 7. Model Selection for RL in Trading

| Model | Pros | Cons | Solution |
|-------|------|------|-----------|
| LSTM based PPO | Captures Sequence Patterns | Sample inefficient, Forget long-term | Experience Replay, Longer Window |
| Transformer based PPO | Captures Global Patterns | High Resource Consumption | Reduce layers/heads |
| ARS (Augmented Random Search) | Very Fast, Simpler | No sequence handling | Combine with LSTM window state |
| GRU based PPO | Light weight sequence handling | Slightly less accurate than LSTM | Good for lower resource setup |

---

## Final Selected Model: PPO with LSTM
Why?
- Market is Sequential
- Pattern memory is essential
- Controlled resource need

---

## 8. Model Architecture Design - PyTorch (No External Library)

### Input → Last 60 OHLCV Candles → Shape (60,6)

### Architecture Layers:
| Layer | Why Needed | Configuration |
|-------|-------------|----------------|
|LSTM Layer| Sequence Learning | Hidden Size=128 |
|FC Layer 1| Feature Compression | 128→64 |
|Actor Head| Action Output | Linear(64,3) |
|Critic Head| Value Prediction | Linear(64,1) |
|Confidence Head| Confidence of Action | Linear(64,1) + Sigmoid |

### Architecture Python Code:
```python
import torch
import torch.nn as nn

class TradingRLModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(TradingRLModel, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)

        self.actor = nn.Linear(64, 3)  # Buy, Sell, Hold
        self.critic = nn.Linear(64, 1)  # V(s)
        self.confidence = nn.Linear(64, 1)  # Sigmoid later

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Last timestep output
        x = torch.relu(self.fc1(x))

        action_logits = self.actor(x)
        state_value = self.critic(x)
        conf_score = torch.sigmoid(self.confidence(x))

        return action_logits, state_value, conf_score
```

---

## 9. Data Preprocessing Strategy

### Why Data Normalization?
- OHLCV values vary hugely (Example: Price can be 100 or 10,000)
- RL models learn better with values between 0 and 1 or small ranges
- Prevents exploding gradients

### Standard Normalization Approach:
```python
Normalized_Value = (Value - Min) / (Max - Min)
```

### Example:
| Raw Close Price | Normalized |
|-----------------|------------|
| 100 | 0.0 |
| 150 | 0.5 |
| 200 | 1.0 |

---

### Custom PyTorch Dataset & Dataloader Code
```python
from torch.utils.data import Dataset
import torch

class TradingDataset(Dataset):
    def __init__(self, data, min_val, max_val):
        self.data = data
        self.min_val = min_val
        self.max_val = max_val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Normalize ohlcv
        norm_ohlcv = (item['state'] - self.min_val) / (self.max_val - self.min_val)

        return {
            'state': torch.tensor(norm_ohlcv, dtype=torch.float),
            'action': torch.tensor(item['action'], dtype=torch.long),
            'reward': torch.tensor(item['reward'], dtype=torch.float),
            'next_state': torch.tensor(item['next_state'], dtype=torch.float),
            'done': torch.tensor(item['done'], dtype=torch.bool),
            'confidence': torch.tensor(item['confidence'], dtype=torch.float)
        }
```

---

## 10. Reward Function Design

### What is Reward in RL?
→ A numeric signal telling the agent how good/bad its action was.

---

### Reward Design for Trading
```python
Reward = Profit - Penalty + Bonus
```

| Term | Meaning | Example |
|------|---------|---------|
|Profit|Real money made |Buy at ₹100 → Sell at ₹105 → Profit=5|
|Penalty|For wrong trade or overtrading| -0.1 |
|Bonus|Correct SL handling or Long holding| +0.2 |

---

### Example Reward Calculation:
#### Buy at ₹100 → Exit at ₹105
```python
Profit = 5
Penalty = -0.1
Bonus = +0.2

Total Reward = 5 - 0.1 + 0.2 = 5.1
```

---

### Python Reward Function Script
```python
def calculate_reward(entry_price, exit_price, penalty, bonus, position_type):
    if position_type == 'BUY':
        profit = exit_price - entry_price
    else:  # SHORT SELL
        profit = entry_price - exit_price

    return profit - penalty + bonus
```

---

## 11. Loss Functions Design & Code

### Why Loss Functions?
- They guide what the model should learn.

---

### PPO Clipped Loss → Stable Policy Update
```python
ratio = new_probs / old_probs
loss = -min(ratio * Advantage, clip(ratio, 1-ε, 1+ε) * Advantage)
```

#### Intuition:
→ No aggressive jumps in policy changes → Stable Learning

---

### Critic Loss → Accurate V(s)
```python
MSE(V(s), Gt)
```

#### Intuition:
→ Teach Critic to estimate true value

---

### Confidence Loss → Correct Reliability
```python
MSE(pred_confidence, normalized(abs(profit)))
```

#### Intuition:
→ High profit → High confidence

---

### Final Combined Loss
```python
Total_Loss = PPO_Loss + c1 * Critic_Loss + c2 * Confidence_Loss + c3 * Entropy_Bonus
```

---

### PyTorch Code for Loss Functions
```python
import torch.nn.functional as F

ppo_loss = torch.min(ratio * advantage, 
                     torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage)
ppo_loss = -ppo_loss.mean()

critic_loss = F.mse_loss(values, returns)
confidence_loss = F.mse_loss(confidences, expected_confidence)

total_loss = ppo_loss + c1 * critic_loss + c2 * confidence_loss - c3 * entropy.mean()
```

---

## 12. Training Environment Setup & Training Loop

### Why Custom Training Environment?
- We are not using OpenAI Gym because:
  - Real stock data
  - Custom Reward design
  - Market-specific constraints

---

### Training Loop Flow:
```
for each epoch:
    for each batch:
        - Get State
        - Get Action from Model
        - Execute Action → Get Reward, Next State
        - Store in Experience Replay

        if enough samples:
            - Sample batch
            - Compute Advantage
            - Calculate PPO Loss, Critic Loss, Confidence Loss
            - Backpropagate Total Loss
```

---

### Python Training Loop Example
```python
for epoch in range(total_epochs):
    for batch in dataloader:
        state = batch['state']
        action = batch['action']
        reward = batch['reward']
        next_state = batch['next_state']

        logits, value, confidence = model(state)
        new_probs = F.softmax(logits, dim=-1)

        # Calculate losses as shown earlier

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

---

## 13. Experience Replay

### Why Experience Replay?
- Stock Market is Slow to change
- New samples are costly
- Need more sample efficiency

### Intuition:
Store all past experience → Randomly sample for training → Prevent forgetting good old trades

---

### Python Code for Experience Replay
```python
import random

class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

---

## 14. Hyperparameters - Selection for Trading RL

| Hyperparameter | Recommended Value | Why |
|----------------|-------------------|-----|
| Discount Factor γ | 0.99 | Stock rewards come after delay |
| PPO Clip ε | 0.2 | Stable learning |
| Learning Rate | 1e-4 | Not too aggressive |
| Trailing SL % | 0.1% to 0.5% | Safe SL range |
| Confidence Threshold | 0.8 | Trade only when sure |
| Buffer Size | 10,000 to 50,000 | Need variety of samples |
| Batch Size | 64 | Memory vs Speed |

### Intuition Behind:
- Higher γ → Long-term reward focused
- ε Clip → Prevent too large policy updates
- Trailing SL % → Depends on market volatility
- Confidence Threshold → Avoid noise trades

---

## 15. Symbols Used in Loss Functions - Explanation (With Real Time Example)

| Symbol | Meaning | Example |
|--------|---------|---------|
| r | Probability ratio of action change | New_prob=0.4, Old_prob=0.3 → r=0.4/0.3=1.33 |
| Aₜ | Advantage → Extra profit expected over baseline | Real reward ₹5, Critic estimated ₹3 → Aₜ=5-3=2 |
| V(sₜ) | Critic predicted future value | Expected profit if stay in current state |
| Gₜ | Discounted Return | Gₜ=Rₜ+γRₜ₊₁+γ²Rₜ₊₂... Example: R=5, γ=0.99 → Gₜ=5+0.99*4+0.99²*3... |
| ε | PPO Clipping Param | Limits ratio r between (1-0.2,1+0.2)=0.8 to 1.2 |
| Confidence | Reliability of action predicted | Output → Confidence=0.85 means trade only if >0.8 |
| Entropy | Exploration force | Entropy high → model not biased to same action |

---

## 16. Evaluation Strategy - Terms, Intuition, Real Time Example & Code

### 1. Win Ratio

#### Meaning: % of Profitable Trades

#### Example:
```python
profitable_trades = 60
total_trades = 100
win_ratio = profitable_trades / total_trades * 100
print(win_ratio)  # Output: 60.0%
```

#### Intuition:
Shows how accurate model is in taking right direction.

---

### 2. Number of Trades

#### Meaning: Total Trade Count

#### Example:
```python
trades_done = len(all_trades)
print(trades_done)  # Output: 120 trades in 1 month
```

#### Intuition:
Too many → Overtrading
Too few → Underutilizing strategy

---

### 3. ROI (Return On Investment)

#### Meaning: % Returns over invested money

#### Example:
```python
initial_capital = 100
final_capital = 120
roi = (final_capital - initial_capital) / initial_capital * 100
print(roi)  # Output: 20%
```

#### Intuition:
Profitability irrespective of number of trades.

---

### 4. Max Drawdown

#### Meaning: Maximum fall from highest peak value

#### Example:
```python
capital_values = [100, 120, 90, 130, 100]
peak = 0
max_dd = 0

for c in capital_values:
    peak = max(peak, c)
    dd = (peak - c) / peak * 100
    max_dd = max(max_dd, dd)

print(max_dd)  # Output: 25.0%
```

#### Intuition:
Risk metric → Lower is safer

---

## 17. Hyperparameter Tuning Strategy with Real Time Evaluation Metrics

### Process:
1. Train model
2. Evaluate using:
    - Win Ratio
    - Number of Trades
    - ROI
    - Max Drawdown

### Example Tuning Decision:
| Issue | Action |
|-------|--------|
|Too many trades| Increase Confidence Threshold|
|Too few trades| Decrease Confidence Threshold|
|High SL Hits| Adjust Trailing SL %|
|Low ROI| Improve Reward structure or reduce penalties|
|High Max Drawdown| Reduce position size or tighten SL|

---



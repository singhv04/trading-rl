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

### Recommended Duration: 
- Minimum 1 Year
- → Best: 2-3 Years

### Each sample of training data: 
- 60 min window
    - takes last 60 min candle data to predict for the next candle
    - analyzes pattern and try to capture the trend
    - more data than this will create much more noise and might confuse as that might not be contributing much
    - less data might limit the pattern capturing and can have insufficient data and very much affected to local noise

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


## 7. Model Selection Overview for Real-Time Trading Bot

### Why Careful Model Selection?
- Stock Market is noisy, volatile & non-stationary.
- Wrong model = Overfitting or Underperformance.
- Need balance between performance & real-time inference speed.

---

## Comparison of Candidate Algorithms

| Algorithm | Pros | Cons | Recommended Usage |
|-----------|------|------|------------------|
| PPO + LSTM | Stable, Proven, Time-series friendly | May miss global patterns | Excellent baseline for trading |
| PPO + Transformer | Captures global dependencies | Slower, resource heavy | Feasible with light design on 9GB Machine |
| ARS | Super Fast, Gradient-free | No sequence learning | Only for highly feature-engineered scalping |
| DDPG | Continuous Action Control | Sensitive, overfits, bad for discrete trading | Avoid for Buy/Sell/Hold setup |

---

## Final Architecture Selected:
> PPO + Conv1D + Transformer Encoder + Stacked LSTM

Reason:
- Conv1D → Extract candle-wise patterns
- Transformer → Capture global sequence relations
- LSTM → Preserve order and memory
- PPO → Stable RL optimizer
- Confidence Head → Avoid random trades

---

## 8. Final Model Architecture Design

### Input:
- Last 60 OHLCV candles → Shape = (60,6)

### Architecture Flow:
```
Input: (Batch, 60, 6)
↓
Conv1D Layer (local pattern smoothing)
↓
Transformer Encoder Layer (light)
↓
2 Layer LSTM (128 → 64)
↓
Fully Connected Layer (64 → 64)
↓
Outputs →
  → Actor Head (3 Neurons) → Buy/Sell/Hold Probabilities
  → Critic Head (1 Neuron) → V(s)
  → Confidence Head (1 Neuron with Sigmoid) → Confidence (0-1)
```

---

## Model Architecture Layerwise Reasoning

| Layer | Purpose | Reason |
|-------|---------|--------|
|Conv1D| Extract small candle patterns| Like Hammer, Doji etc|
|Transformer| Global relation capture | Captures trend shifts |
|LSTM Layer 1| Sequence Memory | Handle time dynamics |
|LSTM Layer 2| Compression & Smoothing | Avoid overfitting |
|FC Layer| Feature fusion | Connect hidden to output |
|Actor Head| Output Action probabilities | Select Buy/Sell/Hold |
|Critic Head| Predict V(s) | Estimate future value |
|Confidence Head| Predict reliability | Select only best trades |

---

## Model Architecture PyTorch Code (Skeleton)

```python
import torch
import torch.nn as nn

class TradingRLModel(nn.Module):
    def __init__(self, input_dim=6, seq_len=60, hidden_dim=128):
        super(TradingRLModel, self).__init__()

        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.actor = nn.Linear(64, 3)
        self.critic = nn.Linear(64, 1)
        self.confidence = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # for Conv1D
        x = torch.relu(self.conv(x))
        x = x.permute(0, 2, 1)  # back to (batch, seq, features)

        x = self.transformer(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]

        x = torch.relu(self.fc1(x))
        
        return self.actor(x), self.critic(x), torch.sigmoid(self.confidence(x))
```

---

## Assuming 8 gb gpu

| Benefit | Why |
|---------|-----|
|Conv1D + Transformer| Lightweight sequence feature enhancer |
|Stacked LSTM| Real trading friendly |
|Separate Heads| Multi-task learning for stability |
|Confidence Prediction| Trade only when model is highly sure |


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

## 10 Updated. Reward Function Design for Real-World Trading Scenario

### Objective:
Reward function should:
- Encourage high profit trades
- Penalize unnecessary trades
- Reward holding strong trend trades
- Discourage bad entries or exits
- Reward risk management like SL usage

---

## Real World Reward Function Formula

```
Reward = λ₁ * Profit Component - λ₂ * Penalty Component + λ₃ * Hold Bonus + λ₄ * SL Usage Bonus
```

Where:
- λ₁ = Weight for Profit (Example: 1)
- λ₂ = Weight for Penalty (Example: 0.1)
- λ₃ = Bonus for Holding profitable trades longer (Example: 0.05 per minute)
- λ₄ = Bonus for using Stop Loss correctly (Example: 0.1)

---

## Profit Component:
```
Profit = Exit Price - Entry Price (BUY)
Profit = Entry Price - Exit Price (SELL)
```

---

## Penalty Component:
Applied when:
- SL hit
- Trade exit in loss
- Overtrading

Fixed or dynamic penalty based on mistake type.

Example: -0.1 per bad exit

---

## Hold Bonus:
Encourage holding longer in clear trend.
```
Hold Bonus = (Holding Duration in minutes) * λ₃
```

---

## SL Usage Bonus:
Reward for respecting Stop Loss execution.
```
SL Bonus = λ₄ (if SL was used properly)
```

---

## Final Reward Calculation Example:
BUY Trade:
- Entry at ₹100
- Exit at ₹105
- Holding for 5 mins
- No SL hit

```
Profit = 105 - 100 = ₹5
Penalty = 0
Hold Bonus = 5 * 0.05 = 0.25
SL Bonus = 0.1

Final Reward = 1*5 - 0.1*0 + 0.25 + 0.1 = ₹5.35
```

---

## Reward Function Python Code
```python
def calculate_reward(entry_price, exit_price, holding_minutes, sl_hit, position_type):
    profit = (exit_price - entry_price) if position_type == 'BUY' else (entry_price - exit_price)

    penalty = 0.1 if (exit_price < entry_price and position_type == 'BUY') or \
                     (exit_price > entry_price and position_type == 'SELL') else 0

    hold_bonus = 0.05 * holding_minutes
    sl_bonus = 0.1 if sl_hit else 0

    total_reward = 1 * profit - 0.1 * penalty + hold_bonus + sl_bonus
    return total_reward
```

---

## Why This Reward Function is Realistic
| Feature | Reason |
|---------|--------|
|Profit based reward| Encourage directionally correct trades |
|Penalty| Avoid bad decisions |
|Hold Bonus| Avoid scalp mentality in trend |
|SL Bonus| Promote risk management |

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

## Symbols Used in Loss Functions - Explanation (With Real Time Example)

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

## 15. Evaluation Strategy - Terms, Intuition, Real Time Example & Code

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


### Example Tuning Decision:
| Issue | Action |
|-------|--------|
|Too many trades| Increase Confidence Threshold|
|Too few trades| Decrease Confidence Threshold|
|High SL Hits| Adjust Trailing SL %|
|Low ROI| Improve Reward structure or reduce penalties|
|High Max Drawdown| Reduce position size or tighten SL|

---



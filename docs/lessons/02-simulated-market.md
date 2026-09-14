# Lesson 2: The Simulated Market

Code: [`mmsim/market.py`](../../mmsim/market.py), [`mmsim/simulator.py`](../../mmsim/simulator.py)

## 1. Why simulate at all?

You can't properly backtest a market maker on historical data, for three reasons:

- **Your quotes change what happens.** Whether you get filled depends on where you sit in the queue
  and who arrives. Historical trades happened without you there.
- **Fills are biased.** In real data you'd be filled mostly when the price is about to move against
  you. A backtest that assumes "I'd have been filled at my quote" ignores exactly that.
- **You never see the true value.** In a simulation you do, so adverse selection can be measured
  exactly rather than guessed.

Real firms use both: simulation to understand mechanisms, and live trading data to calibrate.

## 2. The true value and the consensus

The **true value** `V` follows a random walk. Each step it moves by a normal random amount with
standard deviation σ = 0.6 ticks. Over a 1,000-step session that adds up to about 0.6 × √1000 ≈ 19 ticks.

Most participants don't know `V`. They trade around a **consensus** `c` that catches up slowly:

```
c_t = c_{t-1} + 0.15 × (V_t − c_{t-1})
```

**Worked example.** `V` jumps from 1000 to 1004 and stays there. The gap shrinks by 15% each step:

| Step after jump | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| Consensus | 1000.60 | 1001.11 | 1001.54 | 1002.23 | 1003.21 |

The half-life is ln(0.5) / ln(0.85) ≈ 4.3 steps. **That lag is why informed traders can make money**:
for a few steps, the public price is wrong and they know it.

## 3. The traders

| Type | What they do | Real-world analogue |
|---|---|---|
| **Liquidity providers** | Post limit orders a random number of ticks behind the consensus. Cancel after 20 steps, or right away if the consensus says the order is mispriced | Other market makers, patient institutions |
| **Noise traders** | Market orders on a random side, 1–5 shares | Hedgers, index rebalancing, retail. They trade for reasons unrelated to short-term value |
| **Informed traders** | Know `V`. Buy if the best ask is at least 1 tick below `V`, sell if the best bid is at least 1 tick above | Traders with better models, faster news, or a large order they know is coming |

Arrivals are **Poisson**: each type shows up at an average rate per step (for example 1.5 noise orders), and
each arrival is independent of the last. That's the standard model for "many independent people each
occasionally deciding to trade".

### Why informed traders use IOC limit orders

A market order would keep walking the book, even past `V`, and lose money. An immediate-or-cancel limit at
`V − 1` takes every ask that still leaves at least 1 tick of edge, and stops there.

### Why liquidity providers cancel stale quotes ("quote fading")

In the first version of the simulator they didn't. When the consensus rose, their old sell orders below the
new consensus just sat there. Two things went wrong:

1. The market mid lagged the true value by about 5 ticks on average, far too much.
2. Book imbalance predicted price moves with the opposite sign to real markets.

A diagnostic script traced both problems to the stale orders. Real liquidity providers pull quotes the moment
they look mispriced, so the fix was to cancel any buy above the consensus or sell below it.
**Lesson: check that your simulator reproduces known facts about real markets before you trust
its results.** This makes a good interview story.

## 4. One step, in order

1. The true value moves, and the consensus moves 15% of the way toward it.
2. Liquidity providers' orders expire or get cancelled as stale.
3. The market maker looks at the book (**excluding its own quotes**) and decides where to quote.
4. That step's arrivals happen in random order: liquidity providers post, noise traders and informed traders trade.
5. Inventory and cash are recorded.

## 5. Common random numbers

All randomness for a session (the value path, who arrives, sizes, order) is drawn **before** the session starts,
from the seed. Two strategies run on seed 42 face the *same* market.

Why it matters: session P&L might have a standard deviation of 300 ticks, mostly from market luck. The
**difference** between two strategies on the same seed might only vary by 60, because the luck cancels.
Precision scales with variance, so that's (300/60)² = **25× fewer sessions** for the same confidence.
That's what the paired t-statistics in the results use.

It also guards against a subtle bug. If random numbers were drawn *during* the session, a strategy that
changed the book would change which random numbers later traders got. The two runs would then drift into
different markets.

## Check questions

1. If `consensus_speed` were 1.0, what would happen to informed traders' profits and the market maker's adverse selection?
2. Why do informed traders use an IOC limit order instead of a market order?
3. Noise traders lose money on average. Why do people like them keep trading in real markets?
4. Session P&L has a standard deviation of 400. The paired difference between two strategies has a standard deviation of 80. How many sessions does comparing unpaired averages need, relative to paired, for the same precision?

<details><summary>Answers</summary>

1. The consensus would equal the true value instantly, so liquidity providers would always quote around `V`. Informed traders would rarely find 1 tick of edge, and adverse selection would almost vanish.
2. A market order has no price limit and could buy past the true value. The limit caps the price so every share bought still has at least 1 tick of edge.
3. They aren't trading to predict short-term moves. They're hedging, investing savings, rebalancing, or need cash now. Paying the spread is the price of trading immediately.
4. Roughly (400/80)² = 25× more. (Strictly, comparing two independent averages doubles the variance again, making it about 50×.)

</details>

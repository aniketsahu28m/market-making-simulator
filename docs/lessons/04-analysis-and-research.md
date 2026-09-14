# Lesson 4: Measuring Performance and Signal Research

Code: [`mmsim/analytics.py`](../../mmsim/analytics.py), [`mmsim/research.py`](../../mmsim/research.py),
[`scripts/run_experiments.py`](../../scripts/run_experiments.py). Numbers below come from [`results.md`](../results.md).

## 1. Why total P&L isn't enough

"Strategy A made 1,910 and B made 1,788" tells you *what* happened, not *why*. A trader needs to know which
part of the P&L came from being paid for liquidity, which part was lost to better-informed traders, and which
part was just a position riding the market up or down. Those three call for completely different fixes.

## 2. The three-way P&L decomposition

For a fill with signed quantity `s` (+ for buys, − for sells) at price `p`, where the market maker's
reference mid was `m` and the true value was `V`:

```
spread capture     = s × (m − p)      earned by quoting away from the mid        (≥ 0 by design)
adverse selection  = s × (V − m)      the mid was wrong, in the direction of the trade
inventory          = Σ q_t × (V_{t+1} − V_t)     holding a position while the value moves
```

The first two add up to `s × (V − p)`, the fill's true edge. Adding inventory P&L gives exactly
`cash + final inventory × final value`. A test (`test_pnl_breakdown_adds_up_to_final_pnl`) checks this to 10⁻⁶.

### Worked example

| Step | True value V | Mid m | Event | Spread capture | Adverse selection |
|---|---|---|---|---|---|
| 1 | 100.4 | 100 | Noise trader buys 1 from us at 103 (s = −1) | −1 × (100 − 103) = **+3** | −1 × (100.4 − 100) = **−0.4** |
| 2 | 103.5 | 101 | Informed trader buys 1 from us at 104 (s = −1) | −1 × (101 − 104) = **+3** | −1 × (103.5 − 101) = **−2.5** |
| 3 | 104.0 | | Session ends | | |

Inventory is −1 after step 1 and −2 after step 2, so inventory P&L = (−1)(103.5 − 100.4) + (−2)(104.0 − 103.5) = **−4.1**.

Total = 3 − 0.4 + 3 − 2.5 − 4.1 = **−1.0**. Check directly: cash = 103 + 104 = 207, and we're short 2 at 104 = −208,
so the total is −1.0. ✓

Both fills earned the same spread. The informed one cost 6× more in adverse selection, and the short position it
left behind lost more as the value kept rising.

## 3. What the spread sweep shows

| Half-spread | Mean P&L | Spread capture | Adverse selection | Shares traded |
|---|---|---|---|---|
| 1 | 1,066 | 1,369 | −306 | 1,405 |
| **3** | **1,797** | 2,038 | −241 | 687 |
| 8 | 1,468 | 1,454 | +7 | 182 |

- **At 1 tick**, the market maker trades the most but earns only about 1 tick per share, and adverse selection takes 22% of it.
- **At 8 ticks**, it's rarely hit by informed traders, but volume collapses.
- **3 ticks** is the balance point: the λ(h) × (h − L) trade-off from Lesson 3, measured.

(The slightly *positive* adverse selection at 8 ticks isn't free money. At wide quotes, fills happen disproportionately
when the market maker's side of the book is otherwise empty, and then its reference mid is stale. A good
interview answer is to spot that this is a measurement artifact.)

### More informed traders

| Informed arrivals / step | P&L at half-spread 1 | Adverse sel. ÷ spread capture (h = 1) | Best half-spread |
|---|---|---|---|
| 0.15 | 1,105 | 18% | 3 |
| 0.3 | 1,068 | 23% | 3 |
| 0.6 | 1,005 | 29% | 3 |
| 1.0 | 954 | 35% | 4 |

More informed traders hurt tight quotes badly: adverse selection's share doubles. At 3 ticks there's barely any
change, because informed traders only act with at least 1 tick of edge, and a quote 3 ticks away is rarely that
wrong. The best spread widens only at the highest rate. This is the Glosten–Milgrom intuition, quantified.

## 4. Markouts

A **markout** asks: *h steps after a fill, what is it worth against the mid?*

```
markout(h) = s × (mid_{t+h} − p)      per share
```

Symmetric market maker, half-spread 3:

| Steps after fill | 0 | 1 | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| Noise trader fills | 2.97 | 2.22 | 2.40 | 2.59 | 2.69 | 2.74 | 2.76 |
| Informed trader fills | 2.91 | 1.06 | 0.19 | −1.04 | −1.82 | −2.34 | −2.55 |

- **Informed fills** look just as good as noise fills at the moment of the trade (about +2.9). Within 5 steps they're
  losing money, and they end about **−2.5 ticks per share**. That is adverse selection unfolding in real time.
- **Noise fills** dip at step 1, then recover. The noise trader's own order pushed the mid against us
  (**temporary price impact**), and the liquidity providers then refilled the book around the consensus. This
  is the effect the trade-flow signal exploits in section 6.

Real desks watch markout curves constantly. They're the fastest way to see *who* you're losing to.

## 5. Statistics that hold up

**Standard error and confidence intervals.** Session P&L for Avellaneda–Stoikov has a standard deviation of 206 over
1,000 sessions, so the standard error of the mean is 206/√1000 ≈ 6.5, and the 95% interval is ±1.96 × 6.5 ≈ **±13**.

**Sharpe per session** = mean ÷ std of session P&L. It is *not* annualized: that would need a mapping from sessions
to days, which a simulator doesn't have. Use it to compare strategies here, never to compare with a hedge fund.

**Paired tests with common random numbers.** Avellaneda–Stoikov beat symmetric quoting by +122 ticks per session. Both strategies saw the same
1,000 markets, so we test the 1,000 *differences*. That gives t = 13.6, far more precise than comparing two separate averages
whose shared market luck hasn't cancelled out.

**Separate tuning and evaluation seeds.** If you try 9 values of γ and report the best one's Sharpe *on the same data*,
you'll overstate it, because some of the "best" is luck. That's why every parameter was chosen on tuning seeds and the
final table uses fresh seeds.

It mattered here. The signal strength chosen in tuning showed a tiny Sharpe gain (9.21 vs. 9.14). On fresh
seeds, Sharpe was slightly *lower* (9.15 vs. 9.27), even though P&L rose. Tuning on the test data would have hidden that.

## 6. Signal research

### The candidates

| Signal | Definition | Idea |
|---|---|---|
| Queue imbalance | (bid − ask volume) ÷ total at the best prices | A thin queue on one side is about to be eaten |
| Book imbalance | Same, over 3 levels | Deeper picture of supply and demand |
| Trade flow imbalance | (buyer- − seller-initiated volume) ÷ total, last 10 steps | Who has been aggressive recently |

### Method

1. For each signal and horizon h ∈ {1, 2, 5, 10, 20}, regress the future mid change on the signal: `Δmid = a + b × signal`.
2. Take samples every h steps (**non-overlapping**). Overlapping windows share price moves, so their errors are
   correlated and t-statistics come out inflated.
3. Fit on **training** seeds, score on separate **test** seeds with out-of-sample R², measured against a forecast of zero change.
4. Choose the signal using training data only.

### Results (5-step horizon)

| Signal | Slope | t-stat | Out-of-sample R² |
|---|---|---|---|
| Queue imbalance | +0.07 | 6.3 | 0.0% |
| Book imbalance | −0.64 | −44 | 1.0% |
| **Trade flow imbalance** | **−0.98** | **−57** | **1.6%** |

At a **1-step** horizon, queue imbalance is the best signal, with a *positive* slope (t = 92). That matches real-market studies:
a nearly empty queue gets consumed next.

### Why is trade flow's slope negative?

Recent buying predicts the mid will *fall*. The mechanism is the noise-fill markout from section 4:

1. Noise traders' market orders walk the book and push the mid.
2. Liquidity providers are anchored to a consensus that doesn't react to noise trades, so they refill the book.
3. The mid drifts back.

Price impact has a **permanent** part (information, from informed traders) and a **temporary** part (liquidity
consumption, from everyone). In this market noise flow dominates, so the temporary part wins and the signal
predicts reversals.

**t = −57 but R² = 1.6%. Is that good?** For high-frequency signals, yes. Each prediction explains very little, but
a market maker applies it thousands of times per session. Tiny R² with high significance is typical of real
short-term alpha.

### Using it

The signal market maker shifts its reservation price by `0.5 × (−0.98) × flow_imbalance`. After a burst of
buying it quotes slightly lower, so it sells into the temporary spike and avoids buying at the top.

| vs. plain Avellaneda–Stoikov | Result |
|---|---|
| P&L | +22 ticks / session (paired t = 16.4) |
| Adverse selection per share | −0.364 → −0.333 (**9% less**, paired t = 27) |
| Sharpe | 9.27 → 9.15 (no improvement) |

**Honest conclusion:** the signal adds real but small edge. In this market, most of the risk-adjusted gain comes from
**inventory control**, not prediction.

## 7. Questions an interviewer might ask

**"A Sharpe of 9 is unrealistic."** It's per session in a single-asset simulation with no fees, no latency and no competing
market makers, and the strategy is given the true volatility. The point is the *relative* comparison between
strategies on identical markets.

**"How would you estimate κ and σ for real?"** σ: realized volatility of mid-price changes. κ: record how often
quotes at different distances from the mid get filled, and fit fill rate ∝ e^(−κ × distance). Equivalently, regress
log fill rate on distance.

**"What would you do next?"**
- Add competing market makers, so queue position matters.
- Add fees and rebates, and latency.
- Hedge with a correlated asset.
- Use a fixed Avellaneda–Stoikov horizon instead of time-to-close.
- Rewrite the matching engine in C++ and benchmark it.

**"Why not just backtest on real data?"** Lesson 2, section 1: your own quotes change who trades with you, and historical
data can't show that.

## Check questions

1. A market maker sells 2 shares at 50 when its mid is 49 and the true value is 49.5. What are the spread capture and adverse selection?
2. Informed fills show a markout of +2.9 at step 0 and −2.5 at step 50. What does the curve's shape tell a trader to do?
3. Why take one sample every h steps instead of every step when fitting an h-step prediction?
4. A signal has t = 50 but out-of-sample R² = 1%. Name one reason it could still be valuable, and one reason to be cautious.

<details><summary>Answers</summary>

1. s = −2. Spread capture = −2 × (49 − 50) = **+2**. Adverse selection = −2 × (49.5 − 49) = **−1**.
2. The fills are profitable at first and get picked off over the next few steps. Widen quotes or skew away when the signs of informed
   flow appear, or use a signal that detects it, so fewer of those fills happen.
3. Consecutive h-step windows overlap by h − 1 steps, so their price moves (and errors) are correlated. The regression then treats
   correlated samples as independent and overstates the t-statistic.
4. Valuable: a market maker applies it thousands of times, so a small edge per decision compounds. Cautious: it has to beat trading
   costs, it may be a simulator artifact, and high t-stats with huge samples can reflect tiny, fragile effects.

</details>

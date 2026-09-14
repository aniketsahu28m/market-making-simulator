# Lesson 3: Market-Making Strategies

Code: [`mmsim/strategies.py`](../../mmsim/strategies.py)

## 1. The market maker's three problems

Each step, a market maker picks a bid and an ask. Every choice trades off three things:

1. **Spread vs. fill rate.** Quotes further from the mid earn more per trade but trade less often.
2. **Adverse selection.** Fills are more likely exactly when the quote is wrong (Lesson 1).
3. **Inventory risk.** Every fill leaves a position, and a position loses money when the price moves against it.

The strategies below handle more of these one at a time.

## 2. What the market maker sees

Each step it gets an `Observation`: the step number, its inventory, the best bid and ask, the mid, and three
signals. Everything **excludes its own quotes**.

Why exclude them? Suppose its bid is the best bid. If it computed the mid from a book that includes that bid, then
moving the bid would move the mid, which would move where it wants to bid, and so on. The quotes would
chase their own reflection.

## 3. Strategy 1: symmetric quoting

```
bid = round(mid − h)      ask = round(mid + h)
```

Plus three practical details every real market maker needs:

- **Post-only.** Quotes are clipped so they never cross the book. A market maker that accidentally *takes*
  liquidity pays the spread instead of earning it.
- **Keep queue position.** If this step's price equals last step's, the old order stays in place. Cancelling and
  re-posting would send it to the back of the queue (Lesson 1, question 3).
- **Inventory limit.** At +20 shares it stops bidding; at −20 it stops offering.

### The spread trade-off

Say quotes `h` ticks from the mid get filled at a rate `λ(h)` that falls as `h` grows, and each fill costs
an average `L` to adverse selection. Then expected profit per step is roughly

```
λ(h) × (h − L)
```

At `h` below `L` you lose on every fill. At very large `h` you earn a lot per fill but almost never trade.
The best `h` is somewhere in between, and it moves **wider when more traders are informed**, because `L`
grows. Experiments 1 and 2 in [the results](../results.md) measure both.

### The inventory problem

With inventory `q`, P&L changes by `q × ΔV` each step, so its standard deviation per step is `|q| × σ`. At the
limit (q = 20, σ = 0.6) that's **12 ticks per step**. The market maker earns only a few ticks per share traded.
A symmetric market maker ignores this completely, and in trending sessions it sits pinned at its inventory
limit (see `figures/inventory_paths.png`).

## 4. Strategy 2: Avellaneda–Stoikov (2008)

This is the classic model of optimal market making with inventory risk. Its two key formulas:

```
reservation price    r = mid − q · γ · σ² · τ
total spread         δ = γ · σ² · τ  +  (2/γ) · ln(1 + γ/κ)
bid = r − δ/2        ask = r + δ/2
```

| Symbol | Meaning |
|---|---|
| q | inventory (positive = long) |
| γ | risk aversion: how much you dislike variance |
| σ | volatility per step |
| τ | steps left in the session |
| κ | how fast fill probability falls as you quote further away (fill rate ∝ e^(−κ·distance)) |

**Intuition for `r`.** If you're long, you're already exposed to the price falling, so an extra share is worth
less *to you* than the mid. You move both quotes down: you're more likely to sell and less likely to buy, and
inventory drifts back toward zero.

**Intuition for `δ`.** The first term widens the spread when risk is high: high volatility, a long time left,
high risk aversion. The second term is the spread that trades off fill rate against earnings per fill. As
γ → 0 it tends to 2/κ, so **γ = 0 is exactly the symmetric strategy with h = 1/κ**
(`test_zero_gamma_avellaneda_stoikov_matches_symmetric` checks this).

### Worked example (the project's settings: σ = 0.6, κ = 1/3, γ = 0.0005, start of session τ = 1000)

- Risk term: γσ²τ = 0.0005 × 0.36 × 1000 = **0.18 ticks per share of inventory**
- Liquidity term: (2/0.0005) × ln(1 + 0.0005 × 3) = 4000 × 0.001499 ≈ **5.996**
- Half-spread: (0.18 + 5.996) / 2 ≈ **3.09 ticks**

With the mid at 1000.5:

| Inventory | Reservation price | Bid | Ask |
|---|---|---|---|
| 0 | 1000.5 | round(997.41) = **997** | round(1003.59) = **1004** |
| +10 | 1000.5 − 1.8 = 998.7 | round(995.61) = **996** | round(1001.79) = **1002** |

When long 10 shares, the ask drops 2 ticks (easier to sell) and the bid drops 1 tick (harder to buy).

### Where Avellaneda–Stoikov falls short

- **It has no adverse selection.** The model assumes fills arrive randomly, independent of where the price goes
  next. Our market has informed traders, so that's false, which is why Strategy 3 exists.
- **The end-of-session effect.** As τ → 0 the inventory skew disappears, so near the end the market maker stops caring
  about inventory. Practitioners often use a fixed horizon instead. This is a common interview follow-up.
- **σ and κ must be estimated.** We give it the true σ. A real market maker estimates it from data.

## 5. Strategy 3: adding a short-term signal

```
r = mid − q·γ·σ²·τ  +  β × signal
```

If the signal predicts the mid will rise over the next few steps, both quotes move up. The market maker then sells less
eagerly just before the price rises, which is exactly the fill that would have been adversely selected. Which signal to
use, and how big β should be, is a research question answered in Lesson 4.

## Check questions

1. Using the worked example, where do the quotes go with inventory −10 (short)?
2. What happens to the quotes as γ gets very large? As γ → 0?
3. A market maker re-posts both quotes every step even when the price hasn't changed. What does it lose?
4. Why is taking liquidity especially costly for a market maker's P&L?

<details><summary>Answers</summary>

1. r = 1000.5 + 1.8 = 1002.3, so bid = round(999.21) = 999 and ask = round(1005.39) = 1005. Both move up, so it's more likely to buy back and less likely to sell.
2. Large γ: a huge skew and a wide spread, so it barely trades and flattens inventory aggressively. γ → 0: symmetric quoting with half-spread 1/κ.
3. Queue priority. It goes to the back of the line at its own price every step, so others ahead of it get the fills.
4. It pays the spread instead of earning it, and it usually happens when its fair value is stale, which is itself a form of being adversely selected.

</details>

# Market-Making Simulator

A limit order book exchange simulator for studying market-making strategies:
how a market maker earns the bid-ask spread, how it loses money to better-informed
traders (adverse selection), and how managing inventory changes that trade-off.

## Roadmap

| Stage | Status | Description |
|---|---|---|
| 1. Exchange | ✅ | Price-time priority order book: limit, market and cancel orders |
| 2. Market | ⬜ | Noise traders and informed traders around a hidden "true" price |
| 3. Market maker | ⬜ | Naive symmetric quoting, then inventory-aware quoting (Avellaneda–Stoikov) |
| 4. Analysis | ⬜ | Thousands of sessions; P&L split into spread capture vs. adverse selection; inventory risk |
| 5. Write-up | ⬜ | Results and findings |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## Design notes (stage 1)

- **Integer tick prices.** Prices are stored as integer ticks rather than floats, so
  there are no rounding errors when comparing or matching prices.
- **Price-time priority.** Better prices fill first; at the same price, earlier
  orders fill first (FIFO queue per price level).
- **Trades execute at the resting order's price.** An aggressive buy at 105 against a
  resting sell at 101 trades at 101, so the aggressor gets the price improvement.
- **Market orders are immediate-or-cancel.** Whatever can't fill against the book
  right away is dropped rather than resting.
- **Data structures.** Each side keeps a dict of price → FIFO queue, plus a sorted list
  of active prices. Best bid/ask lookup is O(1); adding or removing a price level is
  O(number of levels), which is small in practice.

# Lesson 1: The Order Book

Code: [`mmsim/order_book.py`](../../mmsim/order_book.py), [`mmsim/orders.py`](../../mmsim/orders.py)

## 1. What an order book is

An exchange keeps a list of people who want to buy and people who want to sell, sorted by price.

```
         ASKS (people selling)
   102 | 5
   101 | 8     ← best ask: the cheapest price you can buy at right now
  ─────────────   spread = 101 − 99 = 2,  mid price = 100
    99 | 10    ← best bid: the highest price you can sell at right now
    98 | 4
         BIDS (people buying)
```

- **Bid**: a price someone will pay. **Ask**: a price someone will accept.
- **Spread** = best ask − best bid. If you buy and immediately sell, the spread is what you lose.
- **Mid price**: halfway between the two. It's the usual guess at what the thing is worth.

A **market maker** keeps a bid and an ask in the book at all times (buy at 99, sell at 101). When one
person sells to it and another buys from it, it earns the 2-point spread.

## 2. Limit orders and market orders

| | Limit order | Market order |
|---|---|---|
| What you say | "Buy 10, but pay no more than 100" | "Buy 10 right now at whatever price" |
| Guarantees | The price | The fill (if anyone is selling) |
| Risk | You might never trade | You might pay a bad price |
| Role | Often waits in the book: a **maker** | Trades immediately: a **taker** |

Makers add liquidity to the book and takers remove it. A market maker earns the spread, but its
orders sit in the book where anyone can trade against them, including people who know more
than it does.

## 3. Matching: price-time priority

1. **Price first.** The best price trades first.
2. **Then time.** At the same price, whoever arrived first fills first. Each price level is a FIFO queue.

Time priority rewards whoever offered liquidity first, so an early place in the queue is valuable.
Some futures markets use *pro-rata* matching instead, where fills are split by order size.

### Worked example (`test_aggressive_order_sweeps_multiple_levels`)

Asks: `100×2 (a)`, `101×2 (b)`, `103×2 (c)`. A **buy limit for 10 @ 101** arrives:

- 2 @ 100 against a, then 2 @ 101 against b
- The next ask is 103, above the buyer's limit, so matching stops
- The remaining 6 **rest as a bid @ 101**

Average price = (200 + 202) / 4 = **100.5**. Going through several price levels like this is
**walking the book**, and the cost it causes is **market impact** (or slippage).

## 4. Trades happen at the resting order's price

A "buy @ 105" against a seller waiting at 101 trades at **101**. The seller committed to that price
first, and the buyer only said *at most* 105.

- **The book can never be crossed.** Any buy priced at or above the best ask trades immediately.
- **There's no gain from overbidding.**

## 5. Market orders are immediate-or-cancel

A market order has no price, so it has no place in a book sorted by price. Whatever can't fill
immediately is dropped. In a book with few orders, a big market order can run up through absurd
prices. Exchanges add price bands to limit this.

## 6. Integer ticks

`0.1 + 0.2 == 0.3` is `False` in floating point. Exchanges avoid this with a **tick size**, the
smallest price step. We store prices as whole numbers of ticks, so comparisons are always exact.

## 7. Data structures and speed

Each side keeps a **dict of price → FIFO queue** (`deque`) and a **sorted list of the active prices**.

| Operation | Cost |
|---|---|
| Best bid or ask | O(1) |
| Add order at an existing price | O(1) |
| Add order at a new price | O(L), where L = number of price levels |
| Each fill while matching | O(1) |
| Cancel | O(k), where k = orders at that price |

`deque` rather than `list`: taking from the front of a `list` is O(n), because every other element
shifts along. A `deque` does it in O(1).

Real low-latency exchanges often use an **array indexed by tick**, `levels[price - min_price]`.
Prices stay close to the current price, so every lookup is O(1) with no sorting.

## 8. Testing with invariants

Besides the hand-written cases, one test sends 5,000 random orders (with a fixed seed) and checks
rules that must always hold: the book is never crossed, filled quantities match the trades,
and no price level has zero quantity. This is **property-based testing**.

## Check questions

1. Asks `51×5, 50×3`, bids `48×4`. A **market sell for 6** arrives. What trades happen, and what does the book look like after?
2. Two buys wait at 100: A (5 shares) first, then B (5 shares). A **sell limit for 7 @ 99** arrives. List the trades and what's left.
3. Why does a market maker care about being first in the queue rather than tenth?
4. Why `deque` rather than `list` for a price level?

<details><summary>Answers</summary>

1. The market sell only hits bids: 4 @ 48. The other 2 are dropped. The asks are untouched and there are no bids left.
2. A: 5 @ **100**, then B: 2 @ **100** (the resting price, not 99). B has 3 left at 100.
3. Fills go to the front of the queue first. Being tenth means you only trade after the first nine are filled, which tends to happen when a large order walks through the level. Those are often the fills you'd rather not get.
4. See section 7: `popleft` is O(1) on a `deque` and O(n) on a `list`.

</details>

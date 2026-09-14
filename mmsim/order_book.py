"""Limit order book with price-time priority matching."""

from __future__ import annotations

import bisect
from collections import deque
from itertools import count

from mmsim.orders import Order, Side, Trade


class OrderBook:
    """Continuous double-auction order book.

    Prices are integer ticks. Resting orders at the same price fill in arrival
    order (FIFO), and trades execute at the resting order's price.
    """

    def __init__(self) -> None:
        self._levels: dict[Side, dict[int, deque[Order]]] = {Side.BUY: {}, Side.SELL: {}}
        # Both lists are sorted ascending: the best bid is last, the best ask is first.
        self._prices: dict[Side, list[int]] = {Side.BUY: [], Side.SELL: []}
        self._resting: dict[int, Order] = {}
        self._order_ids = count(1)
        self._clock = count(1)

    # ---- queries -------------------------------------------------------------

    def best_bid(self) -> int | None:
        prices = self._prices[Side.BUY]
        return prices[-1] if prices else None

    def best_ask(self) -> int | None:
        prices = self._prices[Side.SELL]
        return prices[0] if prices else None

    def mid_price(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def spread(self) -> int | None:
        bid, ask = self.best_bid(), self.best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def depth(self, side: Side, levels: int = 5) -> list[tuple[int, int]]:
        """Top `levels` price levels as (price, total quantity), best price first."""
        prices = self._prices[side]
        ordered = reversed(prices) if side is Side.BUY else iter(prices)
        result = []
        for price in ordered:
            if len(result) == levels:
                break
            result.append((price, sum(o.remaining for o in self._levels[side][price])))
        return result

    def get_order(self, order_id: int) -> Order | None:
        """Return a resting order, or None if it was filled, cancelled or never rested."""
        return self._resting.get(order_id)

    # ---- order entry ---------------------------------------------------------

    def submit_limit(
        self, trader_id: str, side: Side, price: int, quantity: int
    ) -> tuple[Order, list[Trade]]:
        """Match against the book, then rest any unfilled quantity at `price`."""
        if not isinstance(price, int) or price <= 0:
            raise ValueError(f"price must be a positive integer tick, got {price!r}")
        order = self._new_order(trader_id, side, price, quantity)
        trades = self._match(order)
        if order.remaining > 0:
            self._rest(order)
        return order, trades

    def submit_market(
        self, trader_id: str, side: Side, quantity: int
    ) -> tuple[Order, list[Trade]]:
        """Fill as much as possible immediately; any unfilled quantity is dropped."""
        order = self._new_order(trader_id, side, None, quantity)
        return order, self._match(order)

    def cancel(self, order_id: int) -> bool:
        """Cancel a resting order. Returns False if it is no longer resting."""
        order = self._resting.pop(order_id, None)
        if order is None:
            return False
        level = self._levels[order.side][order.price]
        level.remove(order)
        if not level:
            self._remove_level(order.side, order.price)
        return True

    # ---- internals -----------------------------------------------------------

    def _new_order(self, trader_id: str, side: Side, price: int | None, quantity: int) -> Order:
        if not isinstance(quantity, int) or quantity <= 0:
            raise ValueError(f"quantity must be a positive integer, got {quantity!r}")
        return Order(
            order_id=next(self._order_ids),
            trader_id=trader_id,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=next(self._clock),
        )

    def _match(self, order: Order) -> list[Trade]:
        trades = []
        opposite = order.side.opposite
        while order.remaining > 0:
            best = self.best_ask() if order.side is Side.BUY else self.best_bid()
            if best is None or not self._crosses(order, best):
                break
            level = self._levels[opposite][best]
            resting = level[0]
            quantity = min(order.remaining, resting.remaining)
            order.remaining -= quantity
            resting.remaining -= quantity
            trades.append(self._make_trade(order, resting, best, quantity))
            if resting.remaining == 0:
                level.popleft()
                del self._resting[resting.order_id]
                if not level:
                    self._remove_level(opposite, best)
        return trades

    @staticmethod
    def _crosses(order: Order, best_opposite: int) -> bool:
        if order.is_market:
            return True
        if order.side is Side.BUY:
            return best_opposite <= order.price
        return best_opposite >= order.price

    @staticmethod
    def _make_trade(aggressor: Order, resting: Order, price: int, quantity: int) -> Trade:
        buy, sell = (aggressor, resting) if aggressor.side is Side.BUY else (resting, aggressor)
        return Trade(
            buy_order_id=buy.order_id,
            sell_order_id=sell.order_id,
            buyer_id=buy.trader_id,
            seller_id=sell.trader_id,
            price=price,
            quantity=quantity,
            aggressor_side=aggressor.side,
            timestamp=aggressor.timestamp,
        )

    def _rest(self, order: Order) -> None:
        levels = self._levels[order.side]
        if order.price not in levels:
            levels[order.price] = deque()
            bisect.insort(self._prices[order.side], order.price)
        levels[order.price].append(order)
        self._resting[order.order_id] = order

    def _remove_level(self, side: Side, price: int) -> None:
        del self._levels[side][price]
        prices = self._prices[side]
        del prices[bisect.bisect_left(prices, price)]

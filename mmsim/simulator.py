"""Runs one trading session: the market, the background traders and one market maker."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

from mmsim.market import INFORMED, LIQUIDITY, NOISE, Event, MarketConfig, generate_scenario
from mmsim.order_book import OrderBook
from mmsim.orders import Order, Side, Trade
from mmsim.strategies import Observation

MARKET_MAKER = "mm"
IMBALANCE_LEVELS = 3
FLOW_WINDOW = 10  # steps of trade history used for trade flow imbalance


@dataclass(frozen=True)
class Fill:
    """One trade by the market maker."""

    step: int
    side: Side  # the market maker's side
    price: int
    quantity: int
    reference_mid: float  # the mid the market maker observed when it quoted this step
    true_value: float
    counterparty: str
    passive: bool  # True if the market maker's order was resting in the book

    @property
    def signed_quantity(self) -> int:
        return self.quantity if self.side is Side.BUY else -self.quantity


@dataclass
class SessionResult:
    strategy: str
    seed: int
    n_steps: int
    fills: list[Fill]
    # Per-step series. Index t is step t; index 0 is the state before trading starts.
    inventory: list[int]  # at the end of each step
    cash: list[float]  # at the end of each step
    true_values: list[float]
    mids: list[float]  # mid excluding the market maker's quotes, observed at the start of each step
    imbalances: list[float]
    queue_imbalances: list[float]
    flow_imbalances: list[float]

    @property
    def final_pnl(self) -> float:
        """Cash plus the remaining inventory valued at the final true value."""
        return self.cash[-1] + self.inventory[-1] * self.true_values[-1]


def run_session(strategy, config: MarketConfig, seed: int) -> SessionResult:
    scenario = generate_scenario(config, seed)
    book = OrderBook()
    expiries: dict[int, list[int]] = defaultdict(list)
    liquidity_orders: dict[int, Order] = {}
    for side, price, size in scenario.initial_orders:
        order, _ = book.submit_limit(LIQUIDITY, side, price, size)
        expiries[config.liquidity_lifetime].append(order.order_id)
        liquidity_orders[order.order_id] = order

    quotes: dict[Side, Order | None] = {Side.BUY: None, Side.SELL: None}
    inventory = 0
    cash = 0.0
    last_mid = book.mid_price()
    fills: list[Fill] = []
    inventory_path, cash_path = [0], [0.0]
    mids, imbalances, queue_imbalances, flow_imbalances = [last_mid], [0.0], [0.0], [0.0]
    recent_flow: deque[tuple[int, int]] = deque(maxlen=FLOW_WINDOW)  # (signed, total) volume per step

    for step in range(1, config.n_steps + 1):
        true_value = scenario.true_values[step]
        consensus = scenario.consensus_values[step]
        for order_id in expiries.pop(step, ()):
            book.cancel(order_id)
            liquidity_orders.pop(order_id, None)
        _cancel_stale_liquidity(book, liquidity_orders, consensus)

        # Observe the book as if the market maker's own quotes weren't there.
        bids = _levels_excluding_own(book, Side.BUY, quotes[Side.BUY], IMBALANCE_LEVELS)
        asks = _levels_excluding_own(book, Side.SELL, quotes[Side.SELL], IMBALANCE_LEVELS)
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        if best_bid is not None and best_ask is not None:
            last_mid = (best_bid + best_ask) / 2
        imbalance = _imbalance(sum(q for _, q in bids), sum(q for _, q in asks))
        queue_imbalance = _imbalance(bids[0][1] if bids else 0, asks[0][1] if asks else 0)
        flow_total = sum(total for _, total in recent_flow)
        flow_imbalance = sum(signed for signed, _ in recent_flow) / flow_total if flow_total else 0.0
        mids.append(last_mid)
        imbalances.append(imbalance)
        queue_imbalances.append(queue_imbalance)
        flow_imbalances.append(flow_imbalance)

        obs = Observation(
            step=step,
            n_steps=config.n_steps,
            inventory=inventory,
            mid=last_mid,
            best_bid=best_bid,
            best_ask=best_ask,
            imbalance=imbalance,
            queue_imbalance=queue_imbalance,
            flow_imbalance=flow_imbalance,
        )
        quote = strategy.quote(obs)
        bid, ask = _post_only(quote.bid, quote.ask, best_bid, best_ask)
        _update_quotes(book, quotes, {Side.BUY: bid, Side.SELL: ask}, quote.size)

        step_signed_volume = step_volume = 0
        for event in scenario.events[step]:
            if event.kind == LIQUIDITY:
                order = _post_liquidity(book, event, consensus)
                expiries[step + config.liquidity_lifetime].append(order.order_id)
                liquidity_orders[order.order_id] = order
                trades = []
            elif event.kind == NOISE:
                _, trades = book.submit_market(NOISE, event.side, event.size)
            else:
                trades = _informed_trade(book, true_value, event.size, config.informed_threshold)

            for trade in trades:
                step_volume += trade.quantity
                step_signed_volume += trade.quantity if trade.aggressor_side is Side.BUY else -trade.quantity
                fill = _market_maker_fill(trade, step, last_mid, true_value)
                if fill is None:
                    continue
                fills.append(fill)
                inventory += fill.signed_quantity
                cash -= fill.signed_quantity * fill.price

        recent_flow.append((step_signed_volume, step_volume))
        inventory_path.append(inventory)
        cash_path.append(cash)

    return SessionResult(
        strategy=strategy.name,
        seed=seed,
        n_steps=config.n_steps,
        fills=fills,
        inventory=inventory_path,
        cash=cash_path,
        true_values=scenario.true_values,
        mids=mids,
        imbalances=imbalances,
        queue_imbalances=queue_imbalances,
        flow_imbalances=flow_imbalances,
    )


def _imbalance(bid_volume: int, ask_volume: int) -> float:
    total = bid_volume + ask_volume
    return (bid_volume - ask_volume) / total if total else 0.0


def _levels_excluding_own(
    book: OrderBook, side: Side, own: Order | None, levels: int
) -> list[tuple[int, int]]:
    own_price = None
    if own is not None and book.get_order(own.order_id) is own:
        own_price = own.price
    result = []
    for price, quantity in book.depth(side, levels + 1):
        if price == own_price:
            quantity -= own.remaining
        if quantity > 0:
            result.append((price, quantity))
            if len(result) == levels:
                break
    return result


def _post_only(
    bid: int | None, ask: int | None, best_bid: int | None, best_ask: int | None
) -> tuple[int | None, int | None]:
    """Keep quotes from crossing the book, so the market maker never takes liquidity."""
    if bid is not None and best_ask is not None and bid >= best_ask:
        bid = best_ask - 1
    if ask is not None and best_bid is not None and ask <= best_bid:
        ask = best_bid + 1
    if bid is not None and bid < 1:
        bid = None
    if bid is not None and ask is not None and bid >= ask:
        return None, None
    return bid, ask


def _update_quotes(
    book: OrderBook, quotes: dict[Side, Order | None], prices: dict[Side, int | None], size: int
) -> None:
    """Move quotes to the new prices, leaving unchanged quotes alone to keep queue position."""
    to_post = []
    for side, price in prices.items():
        current = quotes[side]
        if current is not None and book.get_order(current.order_id) is not current:
            current = None  # fully filled since last step
        if current is not None and current.price == price:
            continue
        if current is not None:
            book.cancel(current.order_id)
        quotes[side] = None
        if price is not None:
            to_post.append((side, price))

    # Cancel everything stale before posting, so a new quote can't trade with an old one.
    for side, price in to_post:
        order, trades = book.submit_limit(MARKET_MAKER, side, price, size)
        assert not trades, "post-only quote should never trade on arrival"
        quotes[side] = order


def _cancel_stale_liquidity(
    book: OrderBook, liquidity_orders: dict[int, Order], consensus: float
) -> None:
    """Background liquidity providers pull orders their consensus now says are mispriced.

    A buy above the consensus or a sell below it would be a gift to anyone better
    informed, so it's cancelled. This "quote fading" is what makes resting volume
    thin out on the side the price is about to move toward.
    """
    for order_id, order in list(liquidity_orders.items()):
        if book.get_order(order_id) is not order:
            del liquidity_orders[order_id]  # filled
        elif (order.side is Side.BUY and order.price > consensus) or (
            order.side is Side.SELL and order.price < consensus
        ):
            book.cancel(order_id)
            del liquidity_orders[order_id]


def _post_liquidity(book: OrderBook, event: Event, consensus: float) -> Order:
    """Post a background limit order `offset` ticks behind the consensus value.

    Background liquidity is post-only: if the consensus has moved past the book, the
    order joins just behind the opposite best price instead of trading.
    """
    if event.side is Side.BUY:
        price = math.ceil(consensus) - 1 - event.offset
        best_ask = book.best_ask()
        if best_ask is not None:
            price = min(price, best_ask - 1)
    else:
        price = math.floor(consensus) + 1 + event.offset
        best_bid = book.best_bid()
        if best_bid is not None:
            price = max(price, best_bid + 1)
    order, _ = book.submit_limit(LIQUIDITY, event.side, max(price, 1), event.size)
    return order


def _informed_trade(book: OrderBook, true_value: float, size: int, threshold: float) -> list[Trade]:
    """Buy if the best ask is well below the true value, sell if the best bid is well above.

    The order is immediate-or-cancel with a limit, so the informed trader only takes
    prices that still leave at least `threshold` ticks of edge.
    """
    best_ask = book.best_ask()
    if best_ask is not None and true_value - best_ask >= threshold:
        limit = math.floor(true_value - threshold)
        return book.submit_limit(INFORMED, Side.BUY, limit, size, ioc=True)[1]
    best_bid = book.best_bid()
    if best_bid is not None and best_bid - true_value >= threshold:
        limit = math.ceil(true_value + threshold)
        return book.submit_limit(INFORMED, Side.SELL, limit, size, ioc=True)[1]
    return []


def _market_maker_fill(trade: Trade, step: int, reference_mid: float, true_value: float) -> Fill | None:
    if trade.buyer_id == MARKET_MAKER:
        side, counterparty = Side.BUY, trade.seller_id
    elif trade.seller_id == MARKET_MAKER:
        side, counterparty = Side.SELL, trade.buyer_id
    else:
        return None
    return Fill(
        step=step,
        side=side,
        price=trade.price,
        quantity=trade.quantity,
        reference_mid=reference_mid,
        true_value=true_value,
        counterparty=counterparty,
        passive=trade.aggressor_side is not side,
    )

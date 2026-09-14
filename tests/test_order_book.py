import random

import pytest

from mmsim import OrderBook, Side


@pytest.fixture
def book():
    return OrderBook()


def test_empty_book(book):
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.mid_price() is None
    assert book.spread() is None
    assert book.depth(Side.BUY) == []


def test_non_crossing_limits_rest(book):
    _, trades_bid = book.submit_limit("a", Side.BUY, 99, 10)
    _, trades_ask = book.submit_limit("b", Side.SELL, 101, 5)
    assert trades_bid == [] and trades_ask == []
    assert book.best_bid() == 99
    assert book.best_ask() == 101
    assert book.spread() == 2
    assert book.mid_price() == 100


def test_depth_aggregates_and_orders_best_first(book):
    book.submit_limit("a", Side.BUY, 98, 5)
    book.submit_limit("b", Side.BUY, 99, 3)
    book.submit_limit("c", Side.BUY, 99, 4)
    book.submit_limit("d", Side.SELL, 102, 1)
    book.submit_limit("e", Side.SELL, 101, 2)
    assert book.depth(Side.BUY) == [(99, 7), (98, 5)]
    assert book.depth(Side.SELL) == [(101, 2), (102, 1)]
    assert book.depth(Side.BUY, levels=1) == [(99, 7)]


def test_crossing_limit_trades_at_resting_price(book):
    book.submit_limit("seller", Side.SELL, 101, 10)
    order, trades = book.submit_limit("buyer", Side.BUY, 105, 10)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.price == 101
    assert trade.quantity == 10
    assert trade.buyer_id == "buyer" and trade.seller_id == "seller"
    assert trade.aggressor_side is Side.BUY
    assert order.remaining == 0
    assert book.best_ask() is None and book.best_bid() is None


def test_price_priority(book):
    book.submit_limit("worse", Side.SELL, 102, 5)
    book.submit_limit("better", Side.SELL, 101, 5)
    _, trades = book.submit_market("buyer", Side.BUY, 5)
    assert [t.seller_id for t in trades] == ["better"]
    assert book.best_ask() == 102


def test_time_priority_within_level(book):
    book.submit_limit("first", Side.BUY, 100, 5)
    book.submit_limit("second", Side.BUY, 100, 5)
    _, trades = book.submit_market("seller", Side.SELL, 7)
    assert [(t.buyer_id, t.quantity) for t in trades] == [("first", 5), ("second", 2)]
    assert book.depth(Side.BUY) == [(100, 3)]


def test_partial_fill_rests_remainder_at_limit_price(book):
    book.submit_limit("seller", Side.SELL, 100, 4)
    order, trades = book.submit_limit("buyer", Side.BUY, 100, 10)
    assert sum(t.quantity for t in trades) == 4
    assert order.remaining == 6
    assert book.best_bid() == 100
    assert book.best_ask() is None
    assert book.get_order(order.order_id) is order


def test_partially_filled_resting_order_keeps_queue_position(book):
    first, _ = book.submit_limit("first", Side.SELL, 100, 10)
    book.submit_limit("second", Side.SELL, 100, 10)
    book.submit_market("buyer", Side.BUY, 3)
    _, trades = book.submit_market("buyer", Side.BUY, 8)
    assert [(t.seller_id, t.quantity) for t in trades] == [("first", 7), ("second", 1)]
    assert first.remaining == 0


def test_aggressive_order_sweeps_multiple_levels(book):
    book.submit_limit("a", Side.SELL, 100, 2)
    book.submit_limit("b", Side.SELL, 101, 2)
    book.submit_limit("c", Side.SELL, 103, 2)
    order, trades = book.submit_limit("buyer", Side.BUY, 101, 10)
    assert [(t.price, t.quantity) for t in trades] == [(100, 2), (101, 2)]
    assert order.remaining == 6
    assert book.best_bid() == 101
    assert book.best_ask() == 103


def test_market_order_drops_unfilled_quantity(book):
    book.submit_limit("seller", Side.SELL, 100, 3)
    order, trades = book.submit_market("buyer", Side.BUY, 10)
    assert sum(t.quantity for t in trades) == 3
    assert order.remaining == 7
    assert book.best_bid() is None
    assert book.get_order(order.order_id) is None


def test_ioc_limit_fills_up_to_limit_and_drops_rest(book):
    book.submit_limit("a", Side.SELL, 100, 2)
    book.submit_limit("b", Side.SELL, 102, 5)
    order, trades = book.submit_limit("buyer", Side.BUY, 101, 10, ioc=True)
    assert [(t.price, t.quantity) for t in trades] == [(100, 2)]
    assert order.remaining == 8
    assert book.best_bid() is None
    assert book.best_ask() == 102


def test_market_order_on_empty_book(book):
    order, trades = book.submit_market("buyer", Side.BUY, 5)
    assert trades == []
    assert order.remaining == 5


def test_cancel(book):
    order, _ = book.submit_limit("a", Side.BUY, 100, 5)
    other, _ = book.submit_limit("b", Side.BUY, 99, 5)
    assert book.cancel(order.order_id) is True
    assert book.best_bid() == 99
    assert book.get_order(order.order_id) is None
    assert book.cancel(order.order_id) is False
    assert book.cancel(12345) is False
    assert book.get_order(other.order_id) is other


def test_cancel_middle_of_queue(book):
    book.submit_limit("first", Side.SELL, 100, 1)
    middle, _ = book.submit_limit("middle", Side.SELL, 100, 1)
    book.submit_limit("last", Side.SELL, 100, 1)
    book.cancel(middle.order_id)
    _, trades = book.submit_market("buyer", Side.BUY, 2)
    assert [t.seller_id for t in trades] == ["first", "last"]


def test_cannot_cancel_filled_order(book):
    order, _ = book.submit_limit("seller", Side.SELL, 100, 5)
    book.submit_market("buyer", Side.BUY, 5)
    assert book.cancel(order.order_id) is False


@pytest.mark.parametrize("price", [0, -1, 100.5, None])
def test_invalid_price_rejected(book, price):
    with pytest.raises(ValueError):
        book.submit_limit("a", Side.BUY, price, 1)


@pytest.mark.parametrize("quantity", [0, -3, 1.5])
def test_invalid_quantity_rejected(book, quantity):
    with pytest.raises(ValueError):
        book.submit_limit("a", Side.BUY, 100, quantity)
    with pytest.raises(ValueError):
        book.submit_market("a", Side.BUY, quantity)


def test_random_order_flow_invariants():
    """Throw random orders at the book and check properties that must always hold."""
    rng = random.Random(42)
    book = OrderBook()
    filled_by_order: dict[int, int] = {}
    orders = []

    for _ in range(5_000):
        side = rng.choice([Side.BUY, Side.SELL])
        roll = rng.random()
        if roll < 0.1 and orders:
            book.cancel(rng.choice(orders).order_id)
            continue
        if roll < 0.25:
            order, trades = book.submit_market("t", side, rng.randint(1, 20))
        else:
            order, trades = book.submit_limit("t", side, rng.randint(90, 110), rng.randint(1, 20))
        orders.append(order)

        for trade in trades:
            filled_by_order[trade.buy_order_id] = filled_by_order.get(trade.buy_order_id, 0) + trade.quantity
            filled_by_order[trade.sell_order_id] = filled_by_order.get(trade.sell_order_id, 0) + trade.quantity

        bid, ask = book.best_bid(), book.best_ask()
        assert bid is None or ask is None or bid < ask, "book must never be crossed"

    for order in orders:
        assert order.filled == filled_by_order.get(order.order_id, 0)
        assert 0 <= order.remaining <= order.quantity

    for side in Side:
        for price, quantity in book.depth(side, levels=1_000):
            assert quantity > 0

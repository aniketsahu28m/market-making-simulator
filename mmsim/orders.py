"""Order and trade types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Side(Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


# eq=False: two orders are the same only if they are the same object, even if
# their fields happen to match. The book relies on this when removing orders.
@dataclass(eq=False)
class Order:
    order_id: int
    trader_id: str
    side: Side
    price: int | None  # integer ticks; None for market orders
    quantity: int
    timestamp: int
    remaining: int = field(init=False)

    def __post_init__(self) -> None:
        self.remaining = self.quantity

    @property
    def filled(self) -> int:
        return self.quantity - self.remaining

    @property
    def is_market(self) -> bool:
        return self.price is None


@dataclass(frozen=True)
class Trade:
    buy_order_id: int
    sell_order_id: int
    buyer_id: str
    seller_id: str
    price: int
    quantity: int
    aggressor_side: Side
    timestamp: int

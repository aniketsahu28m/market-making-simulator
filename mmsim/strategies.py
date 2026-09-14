"""Market-making strategies: given what the market maker can see, decide where to quote."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    """What the market maker sees at the start of a step. Its own quotes are excluded."""

    step: int
    n_steps: int
    inventory: int
    mid: float
    best_bid: int | None
    best_ask: int | None
    # Short-term signals, each in [-1, 1]; positive means buying pressure.
    imbalance: float  # (bid volume - ask volume) / total over the top 3 price levels
    queue_imbalance: float  # the same, at the best bid and ask only
    flow_imbalance: float  # (buyer- minus seller-initiated volume) / total over recent steps


@dataclass(frozen=True)
class Quote:
    bid: int | None
    ask: int | None
    size: int


def quote_around(
    center: float, half_spread: float, size: int, inventory: int, max_inventory: int
) -> Quote:
    """Quote either side of `center`, rounded to the nearest tick, at least one tick apart.

    At the inventory limit the market maker stops quoting on the side that would
    push its position further.
    """
    bid = math.floor(center - half_spread + 0.5)
    ask = max(math.floor(center + half_spread + 0.5), bid + 1)
    return Quote(
        bid=None if inventory >= max_inventory else bid,
        ask=None if inventory <= -max_inventory else ask,
        size=size,
    )


class SymmetricMaker:
    """Quotes a fixed distance either side of the mid and ignores inventory."""

    def __init__(self, half_spread: float = 1.0, size: int = 2, max_inventory: int = 20):
        self.half_spread = half_spread
        self.size = size
        self.max_inventory = max_inventory

    @property
    def name(self) -> str:
        return f"symmetric(h={self.half_spread:g})"

    def quote(self, obs: Observation) -> Quote:
        return quote_around(obs.mid, self.half_spread, self.size, obs.inventory, self.max_inventory)


class AvellanedaStoikovMaker:
    """Inventory-aware quoting from Avellaneda & Stoikov (2008), in ticks and steps.

        reservation price   r     = mid - q * gamma * sigma^2 * tau
        total spread        delta = gamma * sigma^2 * tau + (2 / gamma) * ln(1 + gamma / kappa)

    q is inventory, tau the steps left in the session, sigma the true value's volatility
    per step, gamma the risk aversion, and kappa how fast the chance of being filled
    falls as a quote moves away from the mid. With gamma = 0 this is symmetric quoting
    with half-spread 1 / kappa.
    """

    def __init__(
        self,
        gamma: float,
        kappa: float,
        volatility: float,
        size: int = 2,
        max_inventory: int = 20,
    ):
        self.gamma = gamma
        self.kappa = kappa
        self.volatility = volatility
        self.size = size
        self.max_inventory = max_inventory

    @property
    def name(self) -> str:
        return f"avellaneda_stoikov(gamma={self.gamma:g}, kappa={self.kappa:g})"

    def reservation_price(self, obs: Observation) -> float:
        tau = obs.n_steps - obs.step
        return obs.mid - obs.inventory * self.gamma * self.volatility**2 * tau

    def half_spread(self, obs: Observation) -> float:
        tau = obs.n_steps - obs.step
        risk_term = self.gamma * self.volatility**2 * tau
        if self.gamma == 0:
            liquidity_term = 2 / self.kappa
        else:
            liquidity_term = (2 / self.gamma) * math.log(1 + self.gamma / self.kappa)
        return (risk_term + liquidity_term) / 2

    def quote(self, obs: Observation) -> Quote:
        return quote_around(
            self.reservation_price(obs),
            self.half_spread(obs),
            self.size,
            obs.inventory,
            self.max_inventory,
        )


class SignalMaker(AvellanedaStoikovMaker):
    """Avellaneda–Stoikov quoting with the reservation price shifted by a short-term signal.

    `feature` names an Observation field that predicts the next move in the mid. Both
    quotes move by `signal_coef` ticks per unit of the signal, so the market maker buys
    less eagerly just before prices fall and sells less eagerly just before they rise.
    """

    def __init__(
        self,
        gamma: float,
        kappa: float,
        volatility: float,
        signal_coef: float,
        feature: str = "flow_imbalance",
        **kwargs,
    ):
        super().__init__(gamma, kappa, volatility, **kwargs)
        self.signal_coef = signal_coef
        self.feature = feature

    @property
    def name(self) -> str:
        return (
            f"signal({self.feature}, gamma={self.gamma:g}, kappa={self.kappa:g}, "
            f"coef={self.signal_coef:.2f})"
        )

    def reservation_price(self, obs: Observation) -> float:
        return super().reservation_price(obs) + self.signal_coef * getattr(obs, self.feature)

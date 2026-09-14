"""The simulated market: a hidden true value and the traders who arrive around it."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mmsim.orders import Side

LIQUIDITY = "liquidity"
NOISE = "noise"
INFORMED = "informed"


@dataclass(frozen=True)
class MarketConfig:
    """Parameters of the simulated market. Prices and volatility are in ticks."""

    n_steps: int = 1_000
    initial_value: float = 1_000.0
    volatility: float = 0.6  # std dev of the true value's change per step

    # Background liquidity providers post limit orders around a consensus value that
    # follows the true value with a lag: each step it closes this fraction of the gap.
    # They cancel orders that the consensus says are mispriced.
    consensus_speed: float = 0.15
    liquidity_rate: float = 2.0  # limit orders per step (Poisson)
    liquidity_offset_p: float = 0.2  # geometric parameter for distance behind the consensus
    liquidity_max_size: int = 5
    liquidity_lifetime: int = 20  # steps before an unfilled order is cancelled
    initial_levels: int = 5

    noise_rate: float = 1.5  # noise market orders per step (Poisson)
    noise_max_size: int = 5

    informed_rate: float = 0.3  # informed trader arrivals per step (Poisson)
    informed_max_size: int = 8
    informed_threshold: float = 1.0  # minimum edge in ticks before an informed trader trades


@dataclass(frozen=True)
class Event:
    kind: str
    side: Side | None  # None for informed traders, who pick a side when they arrive
    offset: int  # liquidity orders only: ticks behind the consensus value
    size: int


@dataclass(frozen=True)
class Scenario:
    """Everything random about one session, drawn before it starts.

    Because the randomness is fixed up front, two strategies run with the same seed
    face the same true-value path and the same arriving traders. Differences in their
    results then come from the strategy rather than from luck.
    """

    true_values: list[float]  # true_values[t] for t = 0..n_steps
    consensus_values: list[float]  # what background liquidity providers believe, same indexing
    events: list[list[Event]]  # events[t] for t = 1..n_steps; events[0] is empty
    initial_orders: list[tuple[Side, int, int]]  # (side, price, size) seeded before step 1


def generate_scenario(config: MarketConfig, seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    n = config.n_steps

    changes = rng.normal(0.0, config.volatility, n)
    true_values = (config.initial_value + np.concatenate([[0.0], np.cumsum(changes)])).tolist()
    consensus_values = [true_values[0]]
    for value in true_values[1:]:
        previous = consensus_values[-1]
        consensus_values.append(previous + config.consensus_speed * (value - previous))

    events: list[list[Event]] = [[] for _ in range(n + 1)]
    arrivals = (
        (LIQUIDITY, config.liquidity_rate, config.liquidity_max_size),
        (NOISE, config.noise_rate, config.noise_max_size),
        (INFORMED, config.informed_rate, config.informed_max_size),
    )
    for kind, rate, max_size in arrivals:
        counts = rng.poisson(rate, n)
        total = int(counts.sum())
        steps = np.repeat(np.arange(1, n + 1), counts).tolist()
        buys = (rng.random(total) < 0.5).tolist()
        offsets = (rng.geometric(config.liquidity_offset_p, total) - 1).tolist()
        sizes = rng.integers(1, max_size + 1, total).tolist()
        for step, is_buy, offset, size in zip(steps, buys, offsets, sizes):
            if kind == INFORMED:
                side = None
            else:
                side = Side.BUY if is_buy else Side.SELL
            events[step].append(Event(kind, side, offset if kind == LIQUIDITY else 0, size))

    # Traders arriving in the same step do so in random order.
    for step_events in events:
        if len(step_events) > 1:
            order = rng.permutation(len(step_events))
            step_events[:] = [step_events[i] for i in order]

    sizes = rng.integers(1, config.liquidity_max_size + 1, 2 * config.initial_levels).tolist()
    initial_orders = []
    for level in range(config.initial_levels):
        initial_orders.append((Side.BUY, math.ceil(config.initial_value) - 1 - level, sizes[2 * level]))
        initial_orders.append((Side.SELL, math.floor(config.initial_value) + 1 + level, sizes[2 * level + 1]))

    return Scenario(
        true_values=true_values,
        consensus_values=consensus_values,
        events=events,
        initial_orders=initial_orders,
    )

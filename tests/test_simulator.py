import numpy as np
import pytest

from mmsim.analytics import aggregate, paired_difference, pnl_breakdown, summarize
from mmsim.market import MarketConfig, generate_scenario
from mmsim.research import FEATURES, ols, out_of_sample_r2, signal_samples
from mmsim.simulator import run_session
from mmsim.strategies import (
    AvellanedaStoikovMaker,
    Observation,
    SignalMaker,
    SymmetricMaker,
)

SMALL = MarketConfig(n_steps=300)
VOL = SMALL.volatility


def make_obs(inventory=0, imbalance=0.0, flow_imbalance=0.0, step=10, n_steps=100):
    return Observation(
        step=step,
        n_steps=n_steps,
        inventory=inventory,
        mid=1000.5,
        best_bid=1000,
        best_ask=1001,
        imbalance=imbalance,
        queue_imbalance=imbalance,
        flow_imbalance=flow_imbalance,
    )


STRATEGIES = [
    SymmetricMaker(half_spread=1.0),
    AvellanedaStoikovMaker(gamma=0.005, kappa=1.0, volatility=VOL),
    SignalMaker(gamma=0.005, kappa=1.0, volatility=VOL, signal_coef=1.0),
]


def test_scenario_is_deterministic():
    a = generate_scenario(SMALL, seed=7)
    b = generate_scenario(SMALL, seed=7)
    c = generate_scenario(SMALL, seed=8)
    assert a == b
    assert a.true_values != c.true_values


def test_session_is_deterministic():
    first = run_session(SymmetricMaker(), SMALL, seed=3)
    second = run_session(SymmetricMaker(), SMALL, seed=3)
    assert first.fills == second.fills
    assert first.inventory == second.inventory


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.name)
def test_pnl_breakdown_adds_up_to_final_pnl(strategy):
    result = run_session(strategy, SMALL, seed=11)
    assert result.fills, "the market maker should trade in a 300-step session"
    assert pnl_breakdown(result).total == pytest.approx(result.final_pnl, abs=1e-6)


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.name)
def test_market_maker_only_provides_liquidity(strategy):
    result = run_session(strategy, SMALL, seed=5)
    assert all(fill.passive for fill in result.fills)


def test_inventory_limit_is_respected():
    strategy = SymmetricMaker(half_spread=0.5, size=2, max_inventory=5)
    config = MarketConfig(n_steps=500, informed_rate=1.0)
    for seed in range(3):
        result = run_session(strategy, config, seed)
        assert max(abs(q) for q in result.inventory) <= strategy.max_inventory + strategy.size


def test_cash_and_inventory_are_consistent_with_fills():
    result = run_session(SymmetricMaker(), SMALL, seed=2)
    assert result.inventory[-1] == sum(f.signed_quantity for f in result.fills)
    assert result.cash[-1] == pytest.approx(-sum(f.signed_quantity * f.price for f in result.fills))


def test_zero_gamma_avellaneda_stoikov_matches_symmetric():
    obs = make_obs(inventory=7)
    as_quote = AvellanedaStoikovMaker(gamma=0.0, kappa=0.5, volatility=VOL).quote(obs)
    assert as_quote == SymmetricMaker(half_spread=2.0).quote(obs)


def test_avellaneda_stoikov_skews_quotes_against_inventory():
    maker = AvellanedaStoikovMaker(gamma=0.05, kappa=1.0, volatility=VOL)
    flat = maker.quote(make_obs(inventory=0))
    long = maker.quote(make_obs(inventory=10))
    short = maker.quote(make_obs(inventory=-10))
    assert long.bid < flat.bid and long.ask < flat.ask
    assert short.bid > flat.bid and short.ask > flat.ask


def test_avellaneda_stoikov_spread_narrows_toward_end_of_session():
    maker = AvellanedaStoikovMaker(gamma=0.05, kappa=1.0, volatility=VOL)
    assert maker.half_spread(make_obs(step=10)) > maker.half_spread(make_obs(step=99))


def test_inventory_limit_stops_quoting_one_side():
    maker = SymmetricMaker(max_inventory=5)
    assert maker.quote(make_obs(inventory=5)).bid is None
    assert maker.quote(make_obs(inventory=-5)).ask is None


@pytest.mark.parametrize("feature", ["imbalance", "flow_imbalance"])
def test_signal_shifts_quotes_toward_buying_pressure(feature):
    base = AvellanedaStoikovMaker(gamma=0.0, kappa=1.0, volatility=VOL)
    signal = SignalMaker(gamma=0.0, kappa=1.0, volatility=VOL, signal_coef=3.0, feature=feature)
    obs = make_obs(**{feature: 0.8})
    assert signal.quote(obs).bid > base.quote(obs).bid
    assert signal.quote(obs).ask > base.quote(obs).ask
    neutral = make_obs()
    assert signal.quote(neutral) == base.quote(neutral)


def test_aggregate_and_paired_difference():
    config = MarketConfig(n_steps=200)
    a = [summarize(run_session(SymmetricMaker(1.0), config, seed)) for seed in range(4)]
    b = [summarize(run_session(SymmetricMaker(2.0), config, seed)) for seed in range(4)]
    stats = aggregate(a)
    assert stats["sessions"] == 4
    assert stats["mean_spread_capture"] + stats["mean_adverse_selection"] + stats[
        "mean_inventory_pnl"
    ] == pytest.approx(stats["mean_pnl"])
    diff = paired_difference(a, b)
    assert diff["n"] == 4
    assert diff["mean_diff"] == pytest.approx(stats["mean_pnl"] - aggregate(b)["mean_pnl"])


def test_ols_recovers_known_slope():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 5_000)
    y = 0.7 * x + rng.normal(0, 0.5, 5_000)
    fit = ols(x, y)
    assert fit.slope == pytest.approx(0.7, abs=0.03)
    assert fit.slope_t_stat > 20
    assert out_of_sample_r2(fit, x, y) > 0.3


@pytest.mark.parametrize("feature", FEATURES)
def test_signal_samples_do_not_overlap(feature):
    result = run_session(SymmetricMaker(), SMALL, seed=1)
    x, y = signal_samples(result, horizon=10, feature=feature)
    assert len(x) == len(y) == len(range(0, SMALL.n_steps - 10, 10))


def test_signals_are_bounded_and_vary():
    result = run_session(SymmetricMaker(), SMALL, seed=4)
    for series in (result.imbalances, result.queue_imbalances, result.flow_imbalances):
        assert all(-1.0 <= value <= 1.0 for value in series)
        assert len(set(series)) > 10

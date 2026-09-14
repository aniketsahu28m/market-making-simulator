"""P&L attribution and performance statistics for market-making sessions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mmsim.market import INFORMED, LIQUIDITY, NOISE
from mmsim.simulator import SessionResult

MARKOUT_HORIZONS = (0, 1, 2, 5, 10, 20, 50)
COUNTERPARTIES = (NOISE, INFORMED, LIQUIDITY)


@dataclass(frozen=True)
class PnLBreakdown:
    spread_capture: float
    adverse_selection: float
    inventory: float

    @property
    def total(self) -> float:
        return self.spread_capture + self.adverse_selection + self.inventory


def pnl_breakdown(result: SessionResult) -> PnLBreakdown:
    """Split session P&L into three parts that add up exactly to the final P&L.

    For a fill with signed quantity s (positive when the market maker buys) at price p,
    where the market maker's reference mid was m and the true value was v:

        spread capture     s * (m - p)   what the quote earned relative to the mid
        adverse selection  s * (v - m)   how far the mid was from the true value, in the
                                         direction of the trade (negative when picked off)

    Together they give s * (v - p), the fill's edge against the true value. The rest of
    the P&L comes from holding inventory while the true value moves:

        inventory          sum over steps of  inventory_t * (v_{t+1} - v_t)
    """
    spread = sum(f.signed_quantity * (f.reference_mid - f.price) for f in result.fills)
    adverse = sum(f.signed_quantity * (f.true_value - f.reference_mid) for f in result.fills)
    values, inventory = result.true_values, result.inventory
    holding = sum(inventory[t] * (values[t + 1] - values[t]) for t in range(result.n_steps))
    return PnLBreakdown(spread_capture=spread, adverse_selection=adverse, inventory=holding)


def summarize(result: SessionResult) -> dict:
    """Per-session statistics, small enough to send back from a worker process."""
    breakdown = pnl_breakdown(result)
    inventory = np.array(result.inventory)
    volume = {cp: 0 for cp in COUNTERPARTIES}
    markouts = {(cp, h): 0.0 for cp in COUNTERPARTIES for h in MARKOUT_HORIZONS}
    last = result.n_steps
    for fill in result.fills:
        volume[fill.counterparty] += fill.quantity
        for h in MARKOUT_HORIZONS:
            future_mid = result.mids[min(fill.step + h, last)] if h else fill.reference_mid
            markouts[fill.counterparty, h] += fill.signed_quantity * (future_mid - fill.price)

    total_volume = sum(volume.values())
    summary = {
        "strategy": result.strategy,
        "seed": result.seed,
        "pnl": result.final_pnl,
        "spread_capture": breakdown.spread_capture,
        "adverse_selection": breakdown.adverse_selection,
        "inventory_pnl": breakdown.inventory,
        "volume": total_volume,
        "n_fills": len(result.fills),
        "mean_abs_inventory": float(np.abs(inventory).mean()),
        "inventory_std": float(inventory.std()),
        "max_abs_inventory": int(np.abs(inventory).max()),
        "final_inventory": int(inventory[-1]),
    }
    for cp in COUNTERPARTIES:
        summary[f"volume_{cp}"] = volume[cp]
        for h in MARKOUT_HORIZONS:
            summary[f"markout_{cp}_{h}"] = markouts[cp, h]
    return summary


def aggregate(summaries: list[dict]) -> dict:
    """Statistics across many sessions of the same strategy."""
    pnl = np.array([s["pnl"] for s in summaries])
    n = len(pnl)
    std = float(pnl.std(ddof=1)) if n > 1 else math.nan
    total_volume = sum(s["volume"] for s in summaries)
    result = {
        "strategy": summaries[0]["strategy"],
        "sessions": n,
        "mean_pnl": float(pnl.mean()),
        "std_pnl": std,
        "sem_pnl": std / math.sqrt(n) if n > 1 else math.nan,
        # Mean over standard deviation of per-session P&L. Not annualized.
        "sharpe_per_session": float(pnl.mean()) / std if std else math.nan,
        "win_rate": float((pnl > 0).mean()),
        "pnl_per_share": float(pnl.sum()) / total_volume if total_volume else math.nan,
        "mean_volume": total_volume / n,
    }
    for key in (
        "spread_capture",
        "adverse_selection",
        "inventory_pnl",
        "mean_abs_inventory",
        "inventory_std",
        "max_abs_inventory",
    ):
        result[f"mean_{key}"] = float(np.mean([s[key] for s in summaries]))
    result["informed_volume_share"] = (
        sum(s[f"volume_{INFORMED}"] for s in summaries) / total_volume if total_volume else math.nan
    )
    for cp in COUNTERPARTIES:
        cp_volume = sum(s[f"volume_{cp}"] for s in summaries)
        result[f"markout_curve_{cp}"] = [
            sum(s[f"markout_{cp}_{h}"] for s in summaries) / cp_volume if cp_volume else math.nan
            for h in MARKOUT_HORIZONS
        ]
    return result


def paired_difference(a: list[dict], b: list[dict], key: str = "pnl") -> dict:
    """Compare two strategies run on the same seeds: mean of (a - b) and its t-statistic.

    Pairing by seed removes the market luck both strategies shared, so the difference
    is measured far more precisely than by comparing the two means separately.
    """
    b_by_seed = {s["seed"]: s[key] for s in b}
    diffs = np.array([s[key] - b_by_seed[s["seed"]] for s in a if s["seed"] in b_by_seed])
    n = len(diffs)
    mean = float(diffs.mean())
    sem = float(diffs.std(ddof=1)) / math.sqrt(n)
    return {"mean_diff": mean, "sem": sem, "t_stat": mean / sem if sem else math.nan, "n": n}

"""Run every experiment and write the figures and results tables.

    .venv/bin/python scripts/run_experiments.py                 # full run
    .venv/bin/python scripts/run_experiments.py --sessions 50   # quick check

Seeds are split into separate blocks so that nothing is tuned and evaluated on the
same simulated data:

    tuning       choose the half-spread, the risk aversion and the signal strength
    research     fit the imbalance signal (train) and test it out of sample (test)
    evaluation   final head-to-head comparison, run once with the chosen settings
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mmsim.analytics import MARKOUT_HORIZONS, aggregate, paired_difference, summarize  # noqa: E402
from mmsim.market import INFORMED, NOISE, MarketConfig  # noqa: E402
from mmsim.research import FEATURES, binned_means, ols, out_of_sample_r2, signal_samples  # noqa: E402
from mmsim.simulator import FLOW_WINDOW, run_session  # noqa: E402
from mmsim.strategies import AvellanedaStoikovMaker, SignalMaker, SymmetricMaker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "docs" / "figures"
RESULTS_MD = ROOT / "docs" / "results.md"
RESULTS_JSON = ROOT / "results" / "summary.json"

HALF_SPREADS = (1, 2, 3, 4, 5, 6, 8)
INFORMED_RATES = (0.15, 0.3, 0.6, 1.0)
GAMMAS = (0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02)
SIGNAL_HORIZONS = (1, 2, 5, 10, 20)
QUOTE_HORIZON = 5  # horizon used to choose a signal and set the signal market maker's coefficient
SIGNAL_SCALES = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
FEATURE_LABELS = {
    "queue_imbalance": "Queue imbalance",
    "imbalance": "Book imbalance (3 levels)",
    "flow_imbalance": "Trade flow imbalance",
}

TUNING_SEED0 = 100_000
RESEARCH_TRAIN_SEED0 = 200_000
RESEARCH_TEST_SEED0 = 300_000
EVALUATION_SEED0 = 0

PNL_PARTS = (
    ("Total P&L", "mean_pnl"),
    ("Spread capture", "mean_spread_capture"),
    ("Adverse selection", "mean_adverse_selection"),
    ("Inventory", "mean_inventory_pnl"),
)

# Validated categorical palette (light mode) and neutral chart chrome.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)


# ---- worker functions (top level so they can run in other processes) -----------


def _summary(strategy, config, seed):
    return summarize(run_session(strategy, config, seed))


def _signal(strategy, config, seed):
    result = run_session(strategy, config, seed)
    return {(f, h): signal_samples(result, h, f) for f in FEATURES for h in SIGNAL_HORIZONS}


def _tracking_error(strategy, config, seed):
    result = run_session(strategy, config, seed)
    mids = np.array(result.mids[1:])
    values = np.array(result.true_values[1:])
    return float(np.abs(mids - values).mean())


def run(pool, strategy, config, seeds) -> list[dict]:
    return list(pool.map(partial(_summary, strategy, config), seeds, chunksize=8))


def log(message: str, start: float) -> None:
    print(f"[{time.time() - start:6.1f}s] {message}", flush=True)


# ---- experiments ------------------------------------------------------------------


def research_signal(pool, strategy, config, n):
    train = list(
        pool.map(
            partial(_signal, strategy, config),
            range(RESEARCH_TRAIN_SEED0, RESEARCH_TRAIN_SEED0 + n),
            chunksize=8,
        )
    )
    test = list(
        pool.map(
            partial(_signal, strategy, config),
            range(RESEARCH_TEST_SEED0, RESEARCH_TEST_SEED0 + n),
            chunksize=8,
        )
    )

    def stack(samples, key):
        return (
            np.concatenate([s[key][0] for s in samples]),
            np.concatenate([s[key][1] for s in samples]),
        )

    stats, fits = {f: {} for f in FEATURES}, {}
    for f in FEATURES:
        for h in SIGNAL_HORIZONS:
            x_train, y_train = stack(train, (f, h))
            x_test, y_test = stack(test, (f, h))
            fit = ols(x_train, y_train)
            fits[f, h] = fit
            stats[f][h] = {
                "slope": fit.slope,
                "t_stat": fit.slope_t_stat,
                "r2_in_sample": fit.r_squared,
                "r2_out_of_sample": out_of_sample_r2(fit, x_test, y_test),
                "ic_out_of_sample": float(np.corrcoef(x_test, y_test)[0, 1]),
                "n_train": fit.n,
                "n_test": len(x_test),
            }

    # Choose using training data only, so the test-set numbers stay honest.
    best = max(FEATURES, key=lambda f: stats[f][QUOTE_HORIZON]["r2_in_sample"])
    x_test, y_test = stack(test, (best, QUOTE_HORIZON))
    return stats, best, fits[best, QUOTE_HORIZON], binned_means(x_test, y_test)


# ---- charts -----------------------------------------------------------------------


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.titlesize": 11,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )


def new_figure(ncols: int, width: float, height: float = 4.0, **kwargs):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), **kwargs)
    return fig, np.atleast_1d(axes)


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=BASELINE, labelcolor=INK_2)


def line_style(color: str) -> dict:
    return dict(
        color=color,
        linewidth=2,
        marker="o",
        markersize=5.5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


def zero_line(ax) -> None:
    ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)


def legend(ax, **kwargs) -> None:
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, **kwargs)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=160)
    plt.close(fig)


def plot_spread_sweep(stats: dict) -> None:
    hs = list(stats)
    fig, axes = new_figure(3, width=14)

    ax = axes[0]
    zero_line(ax)
    for color, (label, key) in zip(SERIES, PNL_PARTS):
        ax.plot(hs, [stats[h][key] for h in hs], label=label, **line_style(color))
    style_axes(ax, "Where the P&L comes from", "Half-spread (ticks)", "Mean per session (ticks)")
    legend(ax)

    ax = axes[1]
    ax.plot(hs, [stats[h]["mean_volume"] for h in hs], **line_style(SERIES[0]))
    ax.set_ylim(bottom=0)
    style_axes(ax, "Shares traded", "Half-spread (ticks)", "Shares per session")

    ax = axes[2]
    ax.plot(hs, [100 * stats[h]["informed_volume_share"] for h in hs], **line_style(SERIES[0]))
    ax.set_ylim(bottom=0)
    style_axes(ax, "Volume from informed traders", "Half-spread (ticks)", "% of shares traded")
    save(fig, "spread_sweep.png")


def plot_informed_sweep(stats: dict) -> None:
    rates = list(stats)
    fig, axes = new_figure(2, width=12)

    ax = axes[0]
    for color, rate in zip(SERIES, rates):
        hs = list(stats[rate])
        ax.plot(hs, [stats[rate][h]["mean_pnl"] for h in hs], label=f"{rate:g} informed arrivals / step",
                **line_style(color))
    style_axes(ax, "Mean P&L by half-spread", "Half-spread (ticks)", "Mean P&L per session (ticks)")
    legend(ax)

    ax = axes[1]
    best = [max(stats[r], key=lambda h: stats[r][h]["mean_pnl"]) for r in rates]
    ax.plot(rates, best, **line_style(SERIES[0]))
    ax.set_ylim(0, max(HALF_SPREADS) + 1)
    style_axes(ax, "Best half-spread", "Informed trader arrivals per step", "Half-spread (ticks)")
    save(fig, "informed_sweep.png")


def plot_gamma_sweep(stats: dict, chosen: float) -> None:
    gammas = list(stats)
    x = np.arange(len(gammas))
    labels = [f"{g:g}" for g in gammas]
    fig, axes = new_figure(3, width=14)

    ax = axes[0]
    std = [stats[g]["std_pnl"] for g in gammas]
    mean = [stats[g]["mean_pnl"] for g in gammas]
    ax.plot(std, mean, color=SERIES[0], linewidth=1, alpha=0.35, zorder=2)
    ax.scatter(std, mean, s=46, color=SERIES[0], edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.margins(x=0.15, y=0.1)
    for g, sx, my in zip(gammas, std, mean):
        if g in (gammas[0], gammas[-1], chosen):
            note = " (chosen)" if g == chosen else ""
            ax.annotate(f"γ = {g:g}{note}", (sx, my), xytext=(7, 5), textcoords="offset points",
                        fontsize=8.5, color=INK_2)
    style_axes(ax, "Risk vs. return", "Std dev of session P&L (ticks)", "Mean P&L per session (ticks)")
    ax.grid(axis="x", color=GRID, linewidth=0.6)

    ax = axes[1]
    ax.plot(x, [stats[g]["mean_inventory_std"] for g in gammas], **line_style(SERIES[0]))
    ax.set_xticks(x, labels)
    ax.set_ylim(bottom=0)
    style_axes(ax, "Inventory risk", "Risk aversion γ", "Std dev of inventory (shares)")

    ax = axes[2]
    ax.plot(x, [stats[g]["sharpe_per_session"] for g in gammas], **line_style(SERIES[0]))
    ax.set_xticks(x, labels)
    style_axes(ax, "Risk-adjusted return", "Risk aversion γ", "Mean ÷ std of session P&L")
    save(fig, "gamma_sweep.png")


def plot_markouts(stats: dict) -> None:
    fig, axes = new_figure(1, width=7)
    ax = axes[0]
    x = np.arange(len(MARKOUT_HORIZONS))
    zero_line(ax)
    for color, (label, cp) in zip(SERIES, (("Noise traders", NOISE), ("Informed traders", INFORMED))):
        ax.plot(x, stats[f"markout_curve_{cp}"], label=label, **line_style(color))
    ax.set_xticks(x, [str(h) for h in MARKOUT_HORIZONS])
    style_axes(ax, "Markouts: what a fill is worth later", "Steps after the fill", "Ticks per share vs. mid")
    legend(ax)
    save(fig, "markouts.png")


def plot_signal(stats: dict, feature: str, fit, binned) -> None:
    fig, axes = new_figure(2, width=13)

    ax = axes[0]
    bx, by = binned
    zero_line(ax)
    ax.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
    xs = np.linspace(bx.min(), bx.max(), 50)
    ax.plot(xs, fit.predict(xs), color=MUTED, linewidth=1.2, zorder=2, label="Fitted on training seeds")
    ax.scatter(bx, by, s=46, color=SERIES[0], edgecolors=SURFACE, linewidths=1.5, zorder=3,
               label="Test seeds, decile means")
    style_axes(ax, f"{FEATURE_LABELS[feature]} vs. next {QUOTE_HORIZON}-step mid change",
               "Signal value", "Mean mid change (ticks)")
    legend(ax, loc="best")

    ax = axes[1]
    x = np.arange(len(SIGNAL_HORIZONS))
    width = 0.8 / len(FEATURES)
    zero_line(ax)
    for i, (color, f) in enumerate(zip(SERIES, FEATURES)):
        values = [100 * stats[f][h]["r2_out_of_sample"] for h in SIGNAL_HORIZONS]
        offset = (i - (len(FEATURES) - 1) / 2) * width
        ax.bar(x + offset, values, width=width * 0.85, color=color, label=FEATURE_LABELS[f])
    ax.set_xticks(x, [str(h) for h in SIGNAL_HORIZONS])
    style_axes(ax, "Out-of-sample R² by signal", "Prediction horizon (steps)", "R² (%)")
    legend(ax, loc="upper left")
    save(fig, "signal_research.png")


def plot_final(final_stats: dict) -> None:
    names = list(final_stats)
    y = np.arange(len(names))[::-1]
    panels = (
        ("Mean P&L per session", lambda s: s["mean_pnl"], lambda s: 1.96 * s["sem_pnl"], "{:,.0f}"),
        ("Sharpe (per session)", lambda s: s["sharpe_per_session"], None, "{:.2f}"),
        ("Adverse selection per share", lambda s: s["mean_adverse_selection"] / s["mean_volume"], None, "{:.2f}"),
        ("Std dev of inventory", lambda s: s["mean_inventory_std"], None, "{:.1f}"),
    )
    fig, axes = new_figure(len(panels), width=16, height=3.6, sharey=True)
    for ax, (title, value, error, fmt) in zip(axes, panels):
        values = [value(final_stats[n]) for n in names]
        errors = [error(final_stats[n]) for n in names] if error else None
        ax.barh(y, values, height=0.5, color=SERIES[0], xerr=errors,
                error_kw=dict(ecolor=INK_2, elinewidth=1, capsize=3))
        ax.axvline(0, color=BASELINE, linewidth=0.8)
        span = max(abs(v) for v in values) or 1
        for i, (yi, v) in enumerate(zip(y, values)):
            tip = v + (errors[i] if errors else 0) * (1 if v >= 0 else -1)  # label past the error bar
            ax.annotate(fmt.format(v), (tip, yi), xytext=(5 if v >= 0 else -5, 0), textcoords="offset points",
                        ha="left" if v >= 0 else "right", va="center", fontsize=8.5, color=INK_2)
        ax.set_xlim(min(0, min(values)) - 0.25 * span, max(0, max(values)) + 0.3 * span)
        style_axes(ax, title, "", "")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    axes[0].set_yticks(y, names)
    save(fig, "final_comparison.png")


def plot_inventory_paths(results: dict) -> None:
    fig, axes = new_figure(1, width=11, height=3.6)
    ax = axes[0]
    zero_line(ax)
    for color, (label, result) in zip(SERIES, results.items()):
        ax.plot(result.inventory, color=color, linewidth=1.3, label=label)
    style_axes(ax, "Inventory through one session (same market for both)", "Step", "Inventory (shares)")
    legend(ax, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    save(fig, "inventory_paths.png")


# ---- report ----------------------------------------------------------------------


def pnl_ci(stats: dict) -> str:
    return f"{stats['mean_pnl']:,.0f} ± {1.96 * stats['sem_pnl']:,.0f}"


def write_markdown(report: dict) -> None:
    cfg = report["config"]
    lines = [
        "# Results",
        "",
        "Generated by `scripts/run_experiments.py`. All P&L is in ticks, marked to the true value at the end",
        f"of each session. {report['sessions']:,} sessions of {cfg['n_steps']:,} steps per strategy in each experiment.",
        "± values are 95% confidence intervals. Sharpe is mean ÷ std of per-session P&L (not annualized).",
        "",
        "## Market",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    lines += [f"| `{k}` | {v} |" for k, v in cfg.items()]
    lines += [
        "",
        f"Average gap between the market mid and the true value: **{report['tracking_error']:.2f} ticks**.",
        "",
        "## 1. Spread sweep: symmetric market maker (tuning seeds)",
        "",
        "![Spread sweep](figures/spread_sweep.png)",
        "",
        "| Half-spread | Mean P&L | Spread capture | Adverse selection | Inventory | Shares/session | Informed share | Sharpe |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h, s in report["spread_sweep"].items():
        lines.append(
            f"| {h} | {pnl_ci(s)} | {s['mean_spread_capture']:,.0f} | {s['mean_adverse_selection']:,.0f} | "
            f"{s['mean_inventory_pnl']:,.0f} | {s['mean_volume']:,.0f} | {100 * s['informed_volume_share']:.1f}% | "
            f"{s['sharpe_per_session']:.2f} |"
        )
    lines += [
        "",
        f"Chosen half-spread (highest mean P&L): **{report['best_half_spread']}** ticks, so κ = 1/{report['best_half_spread']}.",
        "",
        "## 2. How informed traders change the best spread (tuning seeds)",
        "",
        "![Informed sweep](figures/informed_sweep.png)",
        "",
        f"The same spread sweep at different informed trader arrival rates, {report['informed_sessions']:,} sessions each.",
        "",
        "| Informed arrivals per step | Best half-spread | Mean P&L at best | Adverse selection ÷ spread capture | Informed share of volume |",
        "|---|---|---|---|---|",
    ]
    for rate, row in report["informed_sweep"].items():
        best = row["best"]
        lines.append(
            f"| {rate} | {row['best_half_spread']} | {pnl_ci(best)} | "
            f"{100 * -best['mean_adverse_selection'] / best['mean_spread_capture']:.0f}% | "
            f"{100 * best['informed_volume_share']:.1f}% |"
        )
    lines += [
        "",
        "## 3. Risk aversion sweep: Avellaneda–Stoikov (tuning seeds)",
        "",
        "![Gamma sweep](figures/gamma_sweep.png)",
        "",
        "| γ | Mean P&L | Std of P&L | Sharpe | Inventory std | Mean max \\|inventory\\| |",
        "|---|---|---|---|---|---|",
    ]
    for g, s in report["gamma_sweep"].items():
        lines.append(
            f"| {g} | {pnl_ci(s)} | {s['std_pnl']:,.0f} | {s['sharpe_per_session']:.2f} | "
            f"{s['mean_inventory_std']:.1f} | {s['mean_max_abs_inventory']:.1f} |"
        )
    lines += [
        "",
        f"Chosen γ (highest Sharpe): **{report['best_gamma']}**.",
        "",
        "## 4. Research: which signal predicts the mid?",
        "",
        "![Signal research](figures/signal_research.png)",
        "",
        "Three candidate signals, all computed from what the market maker can see (its own quotes excluded):",
        "",
        "- **Queue imbalance**: (bid − ask volume) ÷ total at the best bid and ask",
        "- **Book imbalance**: the same over the top 3 price levels",
        f"- **Trade flow imbalance**: (buyer- − seller-initiated volume) ÷ total over the last {FLOW_WINDOW} steps",
        "",
        "Each signal is regressed on the future mid change: fitted on training seeds, evaluated on separate test",
        "seeds, with non-overlapping windows. The signal is chosen by in-sample R² on the training seeds only.",
        "",
        "| Signal | Horizon (steps) | Slope (ticks) | t-stat (train) | In-sample R² | Out-of-sample R² | Out-of-sample correlation |",
        "|---|---|---|---|---|---|---|",
    ]
    for f, by_horizon in report["signal"].items():
        for h, s in by_horizon.items():
            lines.append(
                f"| {FEATURE_LABELS[f]} | {h} | {s['slope']:+.2f} | {s['t_stat']:.1f} | "
                f"{100 * s['r2_in_sample']:.1f}% | {100 * s['r2_out_of_sample']:.1f}% | {s['ic_out_of_sample']:+.3f} |"
            )
    lines += [
        "",
        f"Chosen signal: **{FEATURE_LABELS[report['best_feature']]}**.",
        "",
        "## 5. Signal strength (tuning seeds)",
        "",
        f"The signal market maker shifts its reservation price by `scale × {report['base_coef']:.2f}` ticks per unit",
        f"of the signal (its fitted {QUOTE_HORIZON}-step slope). Scale 0 is plain Avellaneda–Stoikov.",
        "",
        "| Scale | Mean P&L | Sharpe | Adverse selection per share |",
        "|---|---|---|---|",
    ]
    for scale, s in report["signal_scale_sweep"].items():
        lines.append(
            f"| {scale} | {pnl_ci(s)} | {s['sharpe_per_session']:.2f} | "
            f"{s['mean_adverse_selection'] / s['mean_volume']:.3f} |"
        )
    lines += [
        "",
        f"Chosen scale (highest Sharpe): **{report['best_scale']}**.",
        "",
        "## 6. Final evaluation (fresh seeds, never used for tuning)",
        "",
        "![Final comparison](figures/final_comparison.png)",
        "",
        "| Strategy | Mean P&L | Std of P&L | Sharpe | Win rate | Adverse selection per share | Inventory std |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in report["final"].items():
        lines.append(
            f"| {name} | {pnl_ci(s)} | {s['std_pnl']:,.0f} | {s['sharpe_per_session']:.2f} | "
            f"{100 * s['win_rate']:.1f}% | {s['mean_adverse_selection'] / s['mean_volume']:.3f} | "
            f"{s['mean_inventory_std']:.1f} |"
        )
    lines += [
        "",
        "### Paired comparisons (same seeds, so shared market luck cancels out)",
        "",
        "| Comparison | Metric | Mean difference | t-stat |",
        "|---|---|---|---|",
    ]
    for c in report["comparisons"]:
        lines.append(f"| {c['label']} | {c['metric']} | {c['mean_diff']:+,.2f} | {c['t_stat']:.1f} |")
    lines += [
        "",
        "![Markouts](figures/markouts.png)",
        "",
        "![Inventory paths](figures/inventory_paths.png)",
        "",
    ]
    RESULTS_MD.write_text("\n".join(lines))


# ---- main -------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", type=int, default=1_000, help="sessions per strategy per experiment")
    args = parser.parse_args()
    n = args.sessions

    config = MarketConfig()
    vol = config.volatility
    tuning = range(TUNING_SEED0, TUNING_SEED0 + n)
    evaluation = range(EVALUATION_SEED0, EVALUATION_SEED0 + n)
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    setup_style()
    start = time.time()

    with ProcessPoolExecutor() as pool:
        log("1/6 spread sweep", start)
        spread_stats = {h: aggregate(run(pool, SymmetricMaker(h), config, tuning)) for h in HALF_SPREADS}
        best_h = max(HALF_SPREADS, key=lambda h: spread_stats[h]["mean_pnl"])
        kappa = 1 / best_h

        log("2/6 informed trader sweep", start)
        informed_seeds = range(TUNING_SEED0, TUNING_SEED0 + max(n // 2, 10))
        informed_stats = {
            rate: {
                h: aggregate(run(pool, SymmetricMaker(h), replace(config, informed_rate=rate), informed_seeds))
                for h in HALF_SPREADS
            }
            for rate in INFORMED_RATES
        }

        log("3/6 risk aversion sweep", start)
        gamma_stats = {
            g: aggregate(run(pool, AvellanedaStoikovMaker(g, kappa, vol), config, tuning)) for g in GAMMAS
        }
        best_gamma = max(GAMMAS, key=lambda g: gamma_stats[g]["sharpe_per_session"])

        log("4/6 signal research", start)
        research_maker = AvellanedaStoikovMaker(best_gamma, kappa, vol)
        signal_stats, best_feature, signal_fit, binned = research_signal(pool, research_maker, config, n)
        base_coef = signal_fit.slope

        log("5/6 signal strength sweep", start)
        scale_stats = {
            s: aggregate(
                run(pool, SignalMaker(best_gamma, kappa, vol, s * base_coef, feature=best_feature), config, tuning)
            )
            for s in SIGNAL_SCALES
        }
        best_scale = max(SIGNAL_SCALES, key=lambda s: scale_stats[s]["sharpe_per_session"])

        log("6/6 final evaluation", start)
        symmetric_name = f"Symmetric (h={best_h})"
        as_name = f"Avellaneda–Stoikov (γ={best_gamma:g})"
        signal_name = f"A–S + {FEATURE_LABELS[best_feature].lower()}"
        finalists = {
            "Symmetric, tight (h=1)": SymmetricMaker(1),
            symmetric_name: SymmetricMaker(best_h),
            as_name: AvellanedaStoikovMaker(best_gamma, kappa, vol),
            signal_name: SignalMaker(best_gamma, kappa, vol, best_scale * base_coef, feature=best_feature),
        }
        final_runs = {name: run(pool, strategy, config, evaluation) for name, strategy in finalists.items()}
        tracking_error = float(
            np.mean(list(pool.map(partial(_tracking_error, finalists[as_name], config), range(50))))
        )

    final_stats = {name: aggregate(runs) for name, runs in final_runs.items()}
    comparisons = []
    for label, a, b, metric in (
        (f"{as_name} − {symmetric_name}", as_name, symmetric_name, "pnl"),
        (f"{as_name} − {symmetric_name}", as_name, symmetric_name, "inventory_std"),
        (f"{signal_name} − {as_name}", signal_name, as_name, "pnl"),
        (f"{signal_name} − {as_name}", signal_name, as_name, "adverse_selection"),
    ):
        diff = paired_difference(final_runs[a], final_runs[b], metric)
        comparisons.append({"label": label, "metric": metric, **diff})

    log("writing figures and tables", start)
    plot_spread_sweep(spread_stats)
    plot_informed_sweep(informed_stats)
    plot_gamma_sweep(gamma_stats, best_gamma)
    plot_signal(signal_stats, best_feature, signal_fit, binned)
    plot_markouts(final_stats[symmetric_name])
    plot_final(final_stats)
    plot_inventory_paths({
        symmetric_name: run_session(finalists[symmetric_name], config, EVALUATION_SEED0),
        as_name: run_session(finalists[as_name], config, EVALUATION_SEED0),
    })

    report = {
        "sessions": n,
        "config": {k: v for k, v in vars(config).items()},
        "tracking_error": tracking_error,
        "spread_sweep": {str(h): s for h, s in spread_stats.items()},
        "best_half_spread": best_h,
        "informed_sessions": len(informed_seeds),
        "informed_sweep": {
            f"{rate:g}": {
                "best_half_spread": (bh := max(by_h, key=lambda h: by_h[h]["mean_pnl"])),
                "best": by_h[bh],
                "by_half_spread": {str(h): s for h, s in by_h.items()},
            }
            for rate, by_h in informed_stats.items()
        },
        "gamma_sweep": {f"{g:g}": s for g, s in gamma_stats.items()},
        "best_gamma": best_gamma,
        "signal": {f: {str(h): s for h, s in by_h.items()} for f, by_h in signal_stats.items()},
        "best_feature": best_feature,
        "base_coef": base_coef,
        "signal_scale_sweep": {f"{s:g}": st for s, st in scale_stats.items()},
        "best_scale": best_scale,
        "final": final_stats,
        "comparisons": comparisons,
    }
    RESULTS_JSON.write_text(json.dumps(report, indent=2, default=lambda o: None if isinstance(o, float) and math.isnan(o) else o))
    write_markdown(report)
    log(f"done: {RESULTS_MD.relative_to(ROOT)}, {FIGURES.relative_to(ROOT)}/", start)


if __name__ == "__main__":
    main()

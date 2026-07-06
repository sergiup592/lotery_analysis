"""
EuroMillions ticket toolkit -- honest edition.

One command gives you a complete play plan:

    python3 main.py --jackpot 130000000 --budget 12.50

No module here predicts numbers: for a fair draw that is impossible. The
plan optimises the three levers a fair lottery actually leaves a player:
expected-value steering (fewer co-winners), jackpot timing (EV per euro
varies severalfold), and exact multi-ticket structure (probability mass
moved toward "at least one ticket wins", computed by full enumeration).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.config import PREDICTIONS_DIR, TICKET_COST
from src.ev import BASE_JACKPOT, EVConfig, EVModel, JACKPOT_CAP

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    if getattr(configure_logging, "_configured", False):
        return
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    configure_logging._configured = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run with no flags (or just --jackpot and --budget) to get a complete "
            "play plan: a draw verdict, the tickets to play, and their exact odds. "
            "Nothing here predicts numbers; nothing can."
        ),
    )
    parser.add_argument(
        "--jackpot",
        type=float,
        default=BASE_JACKPOT,
        help="Advertised jackpot in EUR for the draw (default: 17M minimum).",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.0,
        help=f"Budget in EUR; tickets = budget / {TICKET_COST:.2f}. Overrides --tickets.",
    )
    parser.add_argument("--tickets", type=int, default=5, help="Number of tickets to play (default 5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (same seed = same tickets).")
    parser.add_argument(
        "--sales",
        type=float,
        default=0.0,
        help="Europe-wide tickets sold (0 = estimate from cached FDJ winner counts).",
    )
    parser.add_argument(
        "--compare-random",
        type=int,
        default=3,
        help="Random same-budget portfolios to exact-evaluate as a baseline (0 = skip).",
    )
    parser.add_argument(
        "--ev-table",
        action="store_true",
        help="Print expected value per ticket across jackpot levels and exit.",
    )
    parser.add_argument(
        "--validate-ev",
        action="store_true",
        help="Validate the co-winner crowding model against actual FDJ winner counts and exit.",
    )
    parser.add_argument(
        "--calibrate-popularity",
        action="store_true",
        help=(
            "Fit popularity weights from official FDJ winners-per-tier archives "
            "(drop the ZIPs in lottery_data/) and exit."
        ),
    )
    return parser


def build_ev_model(args: argparse.Namespace) -> EVModel:
    return EVModel(
        EVConfig(
            jackpot=max(0.0, float(args.jackpot)),
            sales=float(args.sales) if args.sales else None,
        )
    )


def save_json(payload: Any, output_file: Path) -> None:
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def draw_verdict(ev_model: EVModel) -> list[str]:
    """Plain-language assessment of whether this draw is worth playing."""
    ev_now = ev_model.neutral_ev()["ev_eur"]
    ev_base = EVModel(
        EVConfig(jackpot=BASE_JACKPOT, sales=ev_model.sales),
        main_weights=ev_model.main_weights,
        bonus_weights=ev_model.bonus_weights,
    ).neutral_ev()["ev_eur"]
    per_euro = ev_now / ev_model.config.ticket_price
    ratio = ev_now / ev_base
    jackpot = ev_model.config.jackpot

    lines = [
        f"Advertised jackpot EUR {jackpot:,.0f}; a EUR {ev_model.config.ticket_price:.2f} ticket "
        f"returns ~EUR {ev_now:.2f} in expectation ({per_euro:.0%} back per euro)."
    ]
    if jackpot < 50e6:
        lines.append(
            f"Verdict: LOW-VALUE DRAW ({ratio:.1f}x the minimum-jackpot rate). If you can, "
            f"wait for a rollover above EUR 100M -- the same ticket returns ~2x more per euro there."
        )
    elif jackpot < 150e6:
        lines.append(
            f"Verdict: DECENT ROLLOVER ({ratio:.1f}x the minimum-jackpot rate). "
            f"Meaningfully better than a base draw; bigger rollovers are better still."
        )
    else:
        lines.append(
            f"Verdict: NEAR THE BEST THIS GAME OFFERS ({ratio:.1f}x the minimum-jackpot rate; "
            f"cap is EUR {JACKPOT_CAP:,.0f}). Note: huge jackpots pull in extra players, "
            f"so treat this as an upper bound."
        )
    lines.append(
        "Reality check: expected value stays below the ticket price at every jackpot. "
        "Play for fun, never for profit."
    )
    return lines


def run_play(args: argparse.Namespace) -> None:
    from src.portfolio import build_portfolio, compare_to_random, evaluate_portfolio

    num_tickets = max(1, int(args.budget / TICKET_COST) if args.budget > 0 else args.tickets)
    cost = num_tickets * TICKET_COST
    ev_model = build_ev_model(args)

    records = build_portfolio(num_tickets, ev_model, seed=args.seed)
    tickets = [(tuple(r["main_numbers"]), tuple(r["bonus_numbers"])) for r in records]
    stats = evaluate_portfolio(tickets)
    stats.portfolio_ev_eur = float(sum(r["ev_eur"] for r in records))

    print("================ PLAY PLAN ================")
    for line in draw_verdict(ev_model):
        print(line)
    print()
    print(f"--- Play these {num_tickets} line(s) (EUR {cost:.2f}) ---")
    for index, record in enumerate(records, 1):
        mains = " ".join(f"{n:2d}" for n in record["main_numbers"])
        stars = " ".join(f"{n:2d}" for n in record["bonus_numbers"])
        print(f"  {index}) {mains}  +  stars {stars}")
    print()
    print("--- Why these lines ---")
    avg_crowding = sum(r["jackpot_crowding"] for r in records) / len(records)
    print(
        f"Unpopular combinations: if a line wins, it expects ~{avg_crowding:.1f}x the co-winners "
        f"of an average line (lower is better; jackpot split with fewer people)."
    )
    print(f"Portfolio EV EUR {stats.portfolio_ev_eur:.2f} for EUR {cost:.2f} spent.")
    print()
    print("--- Exact odds for this set (full enumeration, no simulation) ---")
    print(f"P(at least one line wins any prize):   {stats.p_any_prize:.2%}")
    print(f"P(at least one line matches 3+ mains): {stats.p_three_plus_mains:.2%}")
    if args.compare_random > 0:
        baseline = compare_to_random(
            tickets, ev_model=ev_model, n_baselines=args.compare_random, seed=args.seed
        )
        print(
            f"Same budget on random quick-picks:     {baseline['mean_p_any_prize']:.2%} any prize, "
            f"EV EUR {baseline.get('mean_portfolio_ev_eur', float('nan')):.2f}"
        )
    output_file = PREDICTIONS_DIR / "play_plan.json"
    save_json(
        {
            "jackpot": ev_model.config.jackpot,
            "num_tickets": num_tickets,
            "cost_eur": cost,
            "verdict": draw_verdict(ev_model),
            "tickets": records,
            "exact_stats": stats.to_dict(),
        },
        output_file,
    )
    print(f"\nSaved to: {output_file}")
    print(
        "Honesty note: every line has identical odds of winning; these are optimised only "
        "to share less when they win and to spread the set. Long-run cost is unchanged."
    )


def run_ev_table(args: argparse.Namespace) -> None:
    ev_model = build_ev_model(args)
    print("=== EV per ticket vs jackpot (sales held fixed) ===")
    print(f"Sales: {ev_model.sales:,.0f} tickets ({ev_model.sales_source}); ticket price EUR {TICKET_COST:.2f}.")
    for row in ev_model.jackpot_sweep():
        marker = " <- cap" if row["jackpot"] >= JACKPOT_CAP else ""
        print(
            f"  Jackpot EUR {row['jackpot']:>13,.0f}: neutral-ticket EV EUR {row['neutral_ev']:.4f} "
            f"({row['neutral_ev_per_euro']:.3f} per euro){marker}"
        )
    print(
        "Reading: EV varies severalfold with the jackpot but stays below the ticket price -- "
        "'when to play' dominates 'what to pick'."
    )


def run_validate_ev(_: argparse.Namespace) -> None:
    from src.ev import validate_crowding

    report = validate_crowding()
    print("=== Crowding model vs actual FDJ winner counts ===")
    print(f"Draws: {report['n_draws']}  Weights: {report['weights_source']}")
    for tier, metrics in report["per_tier"].items():
        print(
            f"  {tier}: R2 model={metrics['r2_model']:+.3f} vs uniform={metrics['r2_uniform_baseline']:+.3f} "
            f"(mean winners/draw {metrics['mean_winners']:,.0f})"
        )
    print(f"Mean R2 improvement over uniform baseline: {report['mean_r2_improvement_over_uniform']}")
    report_file = PREDICTIONS_DIR / "ev_validation_report.json"
    save_json(report, report_file)
    print(f"Report: {report_file}")


def run_calibrate(_: argparse.Namespace) -> None:
    from src.popularity_fit import FIT_REPORT_FILE, FITTED_WEIGHTS_FILE, run_calibration

    result = run_calibration()
    validation = result.diagnostics.get("validation", {})
    print("=== Popularity Calibration ===")
    print(
        f"Draws used: {result.diagnostics.get('n_draws')} "
        f"({' to '.join(result.diagnostics.get('date_range', []))})"
    )
    print(f"Main-number holdout fit: {validation.get('main_fitted')}")
    print(f"Heuristic prior (same holdout): {validation.get('main_prior_heuristic')}")
    print(f"Star holdout fit: {validation.get('star_fitted')}")
    print(f"Weights: {FITTED_WEIGHTS_FILE}")
    print(f"Report:  {FIT_REPORT_FILE}")


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()

    try:
        if args.calibrate_popularity:
            run_calibrate(args)
        elif args.validate_ev:
            run_validate_ev(args)
        elif args.ev_table:
            run_ev_table(args)
        else:
            run_play(args)
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

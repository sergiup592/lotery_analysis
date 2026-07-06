"""
Economic expected-value engine for EuroMillions tickets.

This replaces heuristic "popularity penalties" with the quantity a player
actually cares about: the expected euro return of a ticket. For a fair draw
the probability of landing in any prize tier is identical for every ticket,
so the *only* thing ticket choice controls is the expected payout
conditional on winning each pari-mutuel tier — i.e. how many co-winners the
ticket is expected to share with. That is priced here, tier by tier.

Model (per ticket T, jackpot J, sales N):

    EV(T) = sum over tiers t of  q_t * fund_t * share(lambda_t(T))

    q_t        exact hypergeometric probability of tier t (same for all T)
    fund_t     jackpot J for tier (5,2); allocation_t * 1.10 EUR * N otherwise
    lambda_t   expected number of *other* winners in tier t given our win
    share(l)   E[1/(1+K)] with K ~ Poisson(l)  =  (1 - exp(-l)) / l

Co-winner intensities reuse exactly the crowding model that
``popularity_fit.py`` calibrates against official FDJ winners-per-tier
archives:

    lambda(k,s) = N * q(k,s) * m_drawn^k * m_undrawn^(5-k)
                             * v_drawn^s * v_undrawn^(2-s)

conditioned on our ticket winning tier (k, s): the draw then contains k of
our mains (a uniform k-subset, so mean weight = our ticket mean) and 5-k
uniform others. For full-main-match tiers (5, s) the exact combination of
mains is pinned to ours, so the delta-method mean is replaced by the exact
product of our numbers' pick weights (normalised by the elementary symmetric
polynomial e_5(w)), times a documented combination-pattern multiplier for
visually patterned lines (1-2-3-4-5 etc.) that per-number weights cannot see.

Prize-structure facts (verified against official EuroMillions rules, 2026):
every participating country pays EUR 1.10 per line into the Common Prize
Fund; tier allocations below; jackpot allocation drops from 50% to 42% from
the 6th consecutive rollover; minimum jackpot EUR 17M; jackpot cap EUR 250M.

Honesty note: EV ranking does not change the odds of winning anything.
For realistic jackpots EV stays well below the EUR 2.50 ticket price; the
engine quantifies *relative* efficiency (EV per euro, ticket vs ticket and
draw vs draw), it does not turn the lottery into a positive-EV game.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import comb, exp
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import (
    BONUS_NUMBER_RANGE,
    MAIN_NUMBER_RANGE,
    N_BONUS,
    N_MAIN,
    TICKET_COST,
)
from . import popularity

# --- Official prize structure -------------------------------------------------

#: Share of the Common Prize Fund allocated to each tier (current rules).
TIER_ALLOCATION: Dict[Tuple[int, int], float] = {
    (5, 2): 0.5000,  # jackpot tier; superseded by the advertised jackpot J
    (5, 1): 0.0261,
    (5, 0): 0.0061,
    (4, 2): 0.0019,
    (4, 1): 0.0035,
    (3, 2): 0.0037,
    (4, 0): 0.0026,
    (2, 2): 0.0130,
    (3, 1): 0.0145,
    (3, 0): 0.0270,
    (1, 2): 0.0327,
    (2, 1): 0.1030,
    (2, 0): 0.1659,
}

CPF_PER_TICKET = 1.10          # EUR paid into the Common Prize Fund per line
BASE_JACKPOT = 17_000_000.0    # guaranteed minimum jackpot
JACKPOT_CAP = 250_000_000.0    # current cap (cycle of 5 draws, then must be won)
FALLBACK_SALES = 25_000_000.0  # used when no winner-count archive is available

# Rule note: from the 6th consecutive rollover the jackpot allocation drops from
# 50% to 42% (difference to reserve). This shifts how the *advertised* jackpot
# accumulates between draws; given an advertised J it does not change this
# draw's EV, so it is deliberately not a model parameter.

# --- Sales elasticity (calibrated on this repo's own winner-count archive) ------
#
# Implied Europe-wide sales across 1,013 draws (2016-2026): ~16-20M tickets on
# base draws, 23.6M median, and 59-67M on the cap-level mega-jackpot draws.
# A linear response calibrated on those anchors keeps high-jackpot EV honest:
# assuming median sales at a EUR 250M jackpot would overstate EV by ~2.5x.
SALES_AT_BASE_JACKPOT = 18_000_000.0
SALES_PER_JACKPOT_EURO = 0.19  # extra tickets sold per EUR of advertised jackpot


def sales_for_jackpot(jackpot: float) -> float:
    """Expected Europe-wide ticket sales for an advertised jackpot."""
    return SALES_AT_BASE_JACKPOT + SALES_PER_JACKPOT_EURO * max(0.0, float(jackpot) - BASE_JACKPOT)

_TOTAL_COMBOS_MAIN = comb(MAIN_NUMBER_RANGE, N_MAIN)
_TOTAL_COMBOS_BONUS = comb(BONUS_NUMBER_RANGE, N_BONUS)


def tier_probability(main_hits: int, star_hits: int) -> float:
    """Exact P(tier) for one ticket against a fair draw (hypergeometric)."""
    main_part = comb(N_MAIN, main_hits) * comb(MAIN_NUMBER_RANGE - N_MAIN, N_MAIN - main_hits)
    star_part = comb(N_BONUS, star_hits) * comb(BONUS_NUMBER_RANGE - N_BONUS, N_BONUS - star_hits)
    return (main_part / _TOTAL_COMBOS_MAIN) * (star_part / _TOTAL_COMBOS_BONUS)


def star_tier_probability(star_hits: int) -> float:
    return comb(N_BONUS, star_hits) * comb(BONUS_NUMBER_RANGE - N_BONUS, N_BONUS - star_hits) / _TOTAL_COMBOS_BONUS


def poisson_share(lam: float) -> float:
    """E[1 / (1 + K)] for K ~ Poisson(lam): the fraction of a pari-mutuel
    tier fund we keep on average, given lam expected co-winners."""
    if lam <= 1e-12:
        return 1.0
    return (1.0 - exp(-lam)) / lam


def elementary_symmetric(weights: Sequence[float], order: int) -> float:
    """e_order(weights) via the standard O(n * order) DP."""
    coefficients = np.zeros(order + 1, dtype=float)
    coefficients[0] = 1.0
    for weight in weights:
        coefficients[1 : order + 1] += weight * coefficients[0:order].copy()
    return float(coefficients[order])


# --- Combination-level pattern multipliers (documented priors) -----------------
#
# Per-number pick weights cannot see that thousands of players share *specific
# visual lines*. These multipliers scale the expected number of co-winners on
# full-main-match tiers only, where the exact line matters. They are
# deliberately conservative priors, not fitted values: exact-line concentration
# is not identifiable from low-tier winner counts (see popularity_fit.py).

def pattern_multiplier(sorted_mains: Sequence[int]) -> float:
    values = list(sorted_mains)
    multiplier = 1.0

    run = _longest_consecutive_run(values)
    if run >= 5:
        multiplier *= 50.0   # 1-2-3-4-5 style lines are massively over-played
    elif run == 4:
        multiplier *= 8.0
    elif run == 3:
        multiplier *= 2.0

    diffs = {b - a for a, b in zip(values, values[1:])}
    if len(diffs) == 1 and run < 5:
        multiplier *= 6.0    # arithmetic progressions (5,10,15,20,25 ...)

    if all(value % 5 == 0 for value in values):
        multiplier *= 5.0    # visual grid columns on the playslip

    if all(value <= 31 for value in values):
        multiplier *= 2.2    # all-calendar lines beyond the per-number effect

    return float(min(multiplier, 200.0))


def _longest_consecutive_run(sorted_values: Sequence[int]) -> int:
    longest = 1
    current = 1
    for previous, value in zip(sorted_values, sorted_values[1:]):
        current = current + 1 if value == previous + 1 else 1
        longest = max(longest, current)
    return longest


# --- Sales estimation -----------------------------------------------------------

def estimate_recent_sales(n_recent: int = 100) -> Optional[float]:
    """
    Median implied Europe-wide ticket sales over the most recent draws, from
    the cached FDJ winner counts: sales ~= total winners / total tier
    probability. Returns None when no cache is available.
    """
    try:
        from .winner_data import RANK_TIERS, read_winner_counts_cache, winners_matrix

        frame = read_winner_counts_cache()
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return None
    if frame.empty:
        return None
    winners = winners_matrix(frame.tail(n_recent))
    total_probability = sum(tier_probability(*tier) for tier in RANK_TIERS.values())
    implied = winners.sum(axis=1) / total_probability
    return float(np.median(implied))


# --- The engine -------------------------------------------------------------------

@dataclass(frozen=True)
class EVConfig:
    jackpot: float = BASE_JACKPOT
    sales: Optional[float] = None          # None -> jackpot-elasticity model
    must_be_won: bool = False              # capped 5th draw: jackpot rolls down if unwon
    raffle_prize: float = 0.0              # country raffle (My Million, Millionaire Maker)
    raffle_pool: float = 0.0               # tickets competing for that raffle
    ticket_price: float = TICKET_COST
    cpf_per_ticket: float = CPF_PER_TICKET

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "EVConfig":
        payload = payload or {}
        sales = payload.get("sales")
        return cls(
            jackpot=max(0.0, float(payload.get("jackpot", BASE_JACKPOT))),
            sales=None if sales in (None, 0, "0", 0.0) else float(sales),
            must_be_won=bool(payload.get("must_be_won", False)),
            raffle_prize=max(0.0, float(payload.get("raffle_prize", 0.0))),
            raffle_pool=max(0.0, float(payload.get("raffle_pool", 0.0))),
            ticket_price=float(payload.get("ticket_price", TICKET_COST)),
            cpf_per_ticket=float(payload.get("cpf_per_ticket", CPF_PER_TICKET)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jackpot": self.jackpot,
            "sales": self.sales,
            "must_be_won": self.must_be_won,
            "raffle_prize": self.raffle_prize,
            "raffle_pool": self.raffle_pool,
            "ticket_price": self.ticket_price,
            "cpf_per_ticket": self.cpf_per_ticket,
        }


class EVModel:
    """
    Prices tickets in euros. Uses the same per-number pick weights as the
    popularity module (fitted from official winner counts when available,
    documented heuristic priors otherwise).
    """

    def __init__(
        self,
        config: EVConfig | None = None,
        main_weights: Optional[np.ndarray] = None,
        bonus_weights: Optional[np.ndarray] = None,
    ):
        self.config = config or EVConfig()
        main = np.asarray(main_weights if main_weights is not None else popularity.MAIN_POPULARITY, dtype=float)
        bonus = np.asarray(bonus_weights if bonus_weights is not None else popularity.BONUS_POPULARITY, dtype=float)
        if main.shape != (MAIN_NUMBER_RANGE,) or bonus.shape != (BONUS_NUMBER_RANGE,):
            raise ValueError("Weight vectors have the wrong length.")
        self.main_weights = main / main.mean()
        self.bonus_weights = bonus / bonus.mean()
        self.weights_source = popularity.weights_provenance() if main_weights is None else "custom"

        if self.config.sales:
            self.sales = float(self.config.sales)
            self.sales_source = "explicit"
        else:
            # Sales respond strongly to the advertised jackpot (16-20M base ->
            # 59-67M at cap in this repo's own archive); pricing high jackpots
            # at median sales would overstate EV, so model the response.
            self.sales = sales_for_jackpot(self.config.jackpot)
            self.sales_source = "jackpot-elasticity model"
        if self.sales <= 0:
            raise ValueError("Ticket sales must be positive.")

        # Normalisers for exact-combination pick probabilities.
        self._e5_main = elementary_symmetric(self.main_weights, N_MAIN)
        self._e2_bonus = elementary_symmetric(self.bonus_weights, N_BONUS)

    # -- crowding ---------------------------------------------------------------

    def _main_mean_crowding(self, mean_ticket_weight: float, k: int) -> float:
        """m_drawn^k * m_undrawn^(5-k) conditional on our ticket matching k mains."""
        rest = (MAIN_NUMBER_RANGE - N_MAIN * mean_ticket_weight) / (MAIN_NUMBER_RANGE - N_MAIN)
        m_drawn = (k * mean_ticket_weight + (N_MAIN - k) * rest) / N_MAIN
        m_undrawn = (MAIN_NUMBER_RANGE - N_MAIN * m_drawn) / (MAIN_NUMBER_RANGE - N_MAIN)
        return (m_drawn ** k) * (m_undrawn ** (N_MAIN - k))

    def _star_mean_crowding(self, mean_ticket_weight: float, s: int) -> float:
        rest = (BONUS_NUMBER_RANGE - N_BONUS * mean_ticket_weight) / (BONUS_NUMBER_RANGE - N_BONUS)
        v_drawn = (s * mean_ticket_weight + (N_BONUS - s) * rest) / N_BONUS
        v_undrawn = (BONUS_NUMBER_RANGE - N_BONUS * v_drawn) / (BONUS_NUMBER_RANGE - N_BONUS)
        return (v_drawn ** s) * (v_undrawn ** (N_BONUS - s))

    def _exact_main_crowding(self, mains: Sequence[int]) -> float:
        """Relative-to-uniform probability that a player line carries exactly
        our five mains: product of pick weights over e_5 mass, times the
        combination-pattern prior."""
        product = float(np.prod([self.main_weights[number - 1] for number in mains]))
        return (product / self._e5_main) * _TOTAL_COMBOS_MAIN * pattern_multiplier(sorted(mains))

    def _exact_bonus_crowding(self, stars: Sequence[int]) -> float:
        product = float(np.prod([self.bonus_weights[number - 1] for number in stars]))
        return (product / self._e2_bonus) * _TOTAL_COMBOS_BONUS

    def cowinner_intensity(self, mains: Sequence[int], stars: Sequence[int], k: int, s: int) -> float:
        """Expected number of other winners in tier (k, s), given our win there."""
        mean_main = float(np.mean([self.main_weights[number - 1] for number in mains]))
        mean_star = float(np.mean([self.bonus_weights[number - 1] for number in stars]))

        if k == N_MAIN:
            main_factor = self._exact_main_crowding(mains) / _TOTAL_COMBOS_MAIN
            if s == N_BONUS:
                star_factor = self._exact_bonus_crowding(stars) / _TOTAL_COMBOS_BONUS
            else:
                star_factor = star_tier_probability(s) * self._star_mean_crowding(mean_star, s)
            return self.sales * main_factor * star_factor

        crowd = self._main_mean_crowding(mean_main, k) * self._star_mean_crowding(mean_star, s)
        return self.sales * tier_probability(k, s) * crowd

    # -- pricing -----------------------------------------------------------------

    def _tier_fund(self, k: int, s: int) -> float:
        if (k, s) == (N_MAIN, N_BONUS):
            return float(self.config.jackpot)
        return TIER_ALLOCATION[(k, s)] * self.config.cpf_per_ticket * self.sales

    def _rolldown_ev(self, our_lambdas: Dict[Tuple[int, int], float]) -> float:
        """
        Must-be-won draws (5th draw at the jackpot cap): if no ticket matches
        5+2, the jackpot is paid to the next prize tier that has winners --
        this has happened (17 Nov 2006: EUR 183M shared by twenty 5+1 winners).
        Our expected euros from that redistribution, walking the official
        prize-rank chain: win tier r AND all higher tiers empty AND share J.
        """
        jackpot = float(self.config.jackpot)
        chain = list(TIER_ALLOCATION)  # official prize-rank order
        ev = 0.0
        cumulative_higher = 0.0  # expected winners across all tiers above rank r
        for index in range(1, len(chain)):
            cumulative_higher += self.sales * tier_probability(*chain[index - 1])
            p_higher_empty = exp(-cumulative_higher)
            if p_higher_empty < 1e-12:
                break
            tier = chain[index]
            ev += tier_probability(*tier) * p_higher_empty * jackpot * poisson_share(our_lambdas[tier])
        return ev

    def _raffle_ev(self) -> float:
        """Country raffles (FDJ My Million, UK Millionaire Maker, ...): a
        guaranteed prize among that country's tickets. Flat EV per ticket;
        better on low-sales (Tuesday) draws."""
        if self.config.raffle_prize > 0 and self.config.raffle_pool > 0:
            return float(self.config.raffle_prize) / float(self.config.raffle_pool)
        return 0.0

    def ticket_ev(self, mains: Sequence[int], stars: Sequence[int], detailed: bool = False) -> Dict[str, Any]:
        mains = [int(value) for value in mains]
        stars = [int(value) for value in stars]

        ev_total = 0.0
        ev_jackpot = 0.0
        tiers: List[Dict[str, float]] = []
        lambda_jackpot = 0.0
        our_lambdas: Dict[Tuple[int, int], float] = {}

        for (k, s) in TIER_ALLOCATION:
            q = tier_probability(k, s)
            lam = self.cowinner_intensity(mains, stars, k, s)
            our_lambdas[(k, s)] = lam
            payout = self._tier_fund(k, s) * poisson_share(lam)
            contribution = q * payout
            ev_total += contribution
            if (k, s) == (N_MAIN, N_BONUS):
                ev_jackpot = contribution
                lambda_jackpot = lam
            if detailed:
                tiers.append(
                    {
                        "tier": f"{k}+{s}",
                        "probability": q,
                        "expected_cowinners": round(lam, 6),
                        "expected_payout_if_win": round(payout, 2),
                        "ev_contribution": round(contribution, 6),
                    }
                )

        ev_rolldown = self._rolldown_ev(our_lambdas) if self.config.must_be_won else 0.0
        ev_raffle = self._raffle_ev()
        ev_total += ev_rolldown + ev_raffle

        result: Dict[str, Any] = {
            "ev_eur": float(ev_total),
            "ev_per_euro": float(ev_total / self.config.ticket_price),
            "ev_jackpot_component": float(ev_jackpot),
            "ev_rolldown_component": float(ev_rolldown),
            "ev_raffle_component": float(ev_raffle),
            "ev_lower_tiers": float(ev_total - ev_jackpot - ev_rolldown - ev_raffle),
            "expected_jackpot_cowinners": float(lambda_jackpot),
            "jackpot_crowding": float(
                lambda_jackpot / (self.sales / (_TOTAL_COMBOS_MAIN * _TOTAL_COMBOS_BONUS))
            ),
        }
        if detailed:
            result["tiers"] = tiers
        return result

    def neutral_ev(self) -> Dict[str, Any]:
        """EV of a popularity-neutral ticket (all crowding factors = 1):
        the fair baseline every real ticket is compared against."""
        ev_total = 0.0
        ev_jackpot = 0.0
        neutral_lambdas: Dict[Tuple[int, int], float] = {}
        for (k, s) in TIER_ALLOCATION:
            q = tier_probability(k, s)
            lam = self.sales * q
            neutral_lambdas[(k, s)] = lam
            contribution = q * self._tier_fund(k, s) * poisson_share(lam)
            ev_total += contribution
            if (k, s) == (N_MAIN, N_BONUS):
                ev_jackpot = contribution
        ev_rolldown = self._rolldown_ev(neutral_lambdas) if self.config.must_be_won else 0.0
        ev_total += ev_rolldown + self._raffle_ev()
        return {
            "ev_eur": float(ev_total),
            "ev_per_euro": float(ev_total / self.config.ticket_price),
            "ev_jackpot_component": float(ev_jackpot),
            "ev_rolldown_component": float(ev_rolldown),
        }

    def attach(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Annotate candidate records in place with EV fields."""
        for candidate in candidates:
            pricing = self.ticket_ev(candidate["main_numbers"], candidate["bonus_numbers"])
            candidate["ev_eur"] = round(pricing["ev_eur"], 6)
            candidate["ev_per_euro"] = round(pricing["ev_per_euro"], 6)
            candidate["expected_jackpot_cowinners"] = round(pricing["expected_jackpot_cowinners"], 4)
            candidate["jackpot_crowding"] = round(pricing["jackpot_crowding"], 4)
        return candidates

    def describe(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "sales_used": self.sales,
            "weights_source": self.weights_source,
            "neutral_ev": self.neutral_ev(),
        }

    # -- portfolio seeding ----------------------------------------------------------

    def anti_popular_candidates(
        self,
        count: int,
        seed: int,
        main_probabilities: np.ndarray,
        bonus_probabilities: np.ndarray,
        source: str,
        beta: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Candidate tickets sampled with probability proportional to the *inverse*
        of pick popularity. Frequency-based pools never explore the unpopular
        corners of the number space; this seeds them so EV selection has
        genuinely cheap-to-share tickets to choose from.
        """
        from .coverage import build_candidate_record  # local import; no cycle at module load

        inv_main = (1.0 / self.main_weights) ** float(beta)
        inv_bonus = (1.0 / self.bonus_weights) ** float(beta)
        inv_main /= inv_main.sum()
        inv_bonus /= inv_bonus.sum()

        rng = np.random.default_rng(seed + 104_729)  # decorrelate from the base pool
        records: List[Dict[str, Any]] = []
        seen = set()
        attempts = max(count * 8, 100)
        while len(records) < count and attempts > 0:
            attempts -= 1
            mains = tuple(sorted(rng.choice(np.arange(1, MAIN_NUMBER_RANGE + 1), size=N_MAIN, replace=False, p=inv_main)))
            stars = tuple(sorted(rng.choice(np.arange(1, BONUS_NUMBER_RANGE + 1), size=N_BONUS, replace=False, p=inv_bonus)))
            key = mains + stars
            if key in seen:
                continue
            seen.add(key)
            records.append(
                build_candidate_record(mains, stars, main_probabilities, bonus_probabilities, source)
            )
        return records

    # -- decision support ------------------------------------------------------------

    def jackpot_sweep(
        self,
        jackpots: Sequence[float] = (BASE_JACKPOT, 50e6, 100e6, 150e6, 200e6, JACKPOT_CAP),
        ticket: Optional[Tuple[Sequence[int], Sequence[int]]] = None,
        include_must_be_won: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        EV per EUR across jackpot levels -- the decision with the largest EV
        range in the game. Sales follow the jackpot-elasticity model per row
        (or stay fixed when explicitly configured). The final row prices the
        capped MUST-BE-WON draw, where the jackpot rolls down if unwon.
        """
        explicit_sales = self.config.sales if self.config.sales else None
        rows: List[Dict[str, Any]] = []
        sweep = [(float(j), False) for j in jackpots]
        if include_must_be_won:
            sweep.append((float(JACKPOT_CAP), True))
        for jackpot, must_be_won in sweep:
            model = EVModel(
                replace(self.config, jackpot=jackpot, sales=explicit_sales, must_be_won=must_be_won),
                main_weights=self.main_weights,
                bonus_weights=self.bonus_weights,
            )
            row: Dict[str, Any] = {
                "jackpot": jackpot,
                "must_be_won": must_be_won,
                "sales": round(model.sales, 0),
                "neutral_ev": round(model.neutral_ev()["ev_eur"], 4),
            }
            if ticket is not None:
                row["ticket_ev"] = round(model.ticket_ev(ticket[0], ticket[1])["ev_eur"], 4)
            row["neutral_ev_per_euro"] = round(row["neutral_ev"] / self.config.ticket_price, 4)
            rows.append(row)
        return rows


# --- Validation against actual winner counts ---------------------------------------

def validate_crowding(n_recent: Optional[int] = None, frame=None) -> Dict[str, Any]:
    """
    Test the crowding model on real draws: for every draw, predict the *share*
    of total winners landing in each tier (anchor-free: sales cancel), compare
    against actual shares, and report per-tier R^2 next to a uniform-model
    baseline (all crowding factors = 1). Uses the same weights the EV engine
    prices with.
    """
    from .winner_data import RANK_TIERS, read_winner_counts_cache, winners_matrix

    if frame is None:
        frame = read_winner_counts_cache()
    if n_recent:
        frame = frame.tail(n_recent).reset_index(drop=True)
    winners = winners_matrix(frame)

    model = EVModel(EVConfig(sales=FALLBACK_SALES))  # sales cancel in shares
    ranks = list(RANK_TIERS)
    tiers = [RANK_TIERS[rank] for rank in ranks]
    q = np.asarray([tier_probability(*tier) for tier in tiers])

    def draw_crowd(mains: Sequence[int], stars: Sequence[int]) -> np.ndarray:
        mean_main = float(np.mean([model.main_weights[number - 1] for number in mains]))
        mean_star = float(np.mean([model.bonus_weights[number - 1] for number in stars]))
        factors = []
        for (k, s) in tiers:
            factors.append(
                model._main_mean_crowding(mean_main, k) * model._star_mean_crowding(mean_star, s)
            )
        return np.asarray(factors)

    predicted_share = np.zeros_like(winners)
    uniform_share = np.tile(q / q.sum(), (len(frame), 1))
    for idx, (mains, stars) in enumerate(zip(frame["main_numbers"], frame["bonus_numbers"])):
        weighted = q * draw_crowd(mains, stars)
        predicted_share[idx] = weighted / weighted.sum()

    actual_total = winners.sum(axis=1, keepdims=True)
    actual_share = winners / np.clip(actual_total, 1.0, None)

    per_tier: Dict[str, Dict[str, float]] = {}
    for column, rank in enumerate(ranks):
        actual = actual_share[:, column]
        if winners[:, column].mean() < 20:  # Poisson noise dominates tiny tiers
            continue
        mask = actual > 0
        log_actual = np.log(actual[mask])
        log_model = np.log(predicted_share[mask][:, column])
        log_uniform = np.log(uniform_share[mask][:, column])
        baseline_var = float(np.sum((log_actual - log_actual.mean()) ** 2))
        if baseline_var <= 0:
            continue
        r2_model = 1.0 - float(np.sum((log_actual - log_model) ** 2)) / baseline_var
        r2_uniform = 1.0 - float(np.sum((log_actual - log_uniform) ** 2)) / baseline_var
        per_tier[f"{RANK_TIERS[rank][0]}+{RANK_TIERS[rank][1]}"] = {
            "n_draws": int(mask.sum()),
            "r2_model": round(r2_model, 4),
            "r2_uniform_baseline": round(r2_uniform, 4),
            "mean_winners": round(float(winners[:, column].mean()), 1),
        }

    improvements = [
        metrics["r2_model"] - metrics["r2_uniform_baseline"] for metrics in per_tier.values()
    ]
    return {
        "n_draws": int(len(frame)),
        "weights_source": model.weights_source,
        "per_tier": per_tier,
        "mean_r2_improvement_over_uniform": round(float(np.mean(improvements)), 4) if improvements else None,
    }

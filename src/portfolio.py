"""
Exact portfolio win-structure evaluation and construction.

Prediction is impossible for a fair draw, and per-ticket EV steering
(``ev.py``) already prices prize sharing. The one remaining lever a
multi-ticket player controls is the *joint* structure of their tickets: for
the same budget, overlapping tickets win in clumps (usually nothing, sometimes
several small prizes at once) while spread-out tickets maximise the chance
that at least one of them wins something. Expected value of hits is identical
either way -- only the shape of the outcome distribution moves. This module
makes that shape exact instead of heuristic.

Method: enumerate *all* C(50,5) = 2,118,760 possible main draws as 64-bit
masks and compute every ticket's match count against every possible draw
(vectorised popcount). Lucky stars are folded in analytically per main draw:
a ticket wins a prize iff it matches >= 2 mains (tier 2+0 exists), or exactly
1 main with both stars (tier 1+2). No sampling, no approximation -- the
reported probabilities are exact to floating point.

Two structural facts this module surfaces honestly:

1. For T tickets with pairwise-disjoint mains (T*5 <= 50), the main-side win
   structure is *invariant* to which numbers you choose: any two disjoint
   portfolios are relabelings of each other, and the uniform draw is
   permutation-invariant. Disjointness buys the structure; the remaining
   freedom (which disjoint numbers, which star pairs) is pure EV -- so we
   spend it on unpopular numbers.
2. E[number of winning tickets] = T * P(single ticket wins) for *any*
   portfolio (linearity of expectation). No construction changes the average;
   diversification only reallocates probability mass toward "at least one".

References: covering-design literature (La Jolla Covering Repository;
Cushing & Stewart 2023, "You need 27 tickets to guarantee a win on the UK
National Lottery") establishes guarantee-style constructions; famously, the
guaranteed win usually pays less than the tickets cost. This module reports
exact probabilities and euros instead of chasing guarantees.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from math import comb
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import BONUS_NUMBER_RANGE, MAIN_NUMBER_RANGE, N_BONUS, N_MAIN

Ticket = Tuple[Tuple[int, ...], Tuple[int, ...]]

_TOTAL_MAIN_DRAWS = comb(MAIN_NUMBER_RANGE, N_MAIN)      # 2,118,760
_TOTAL_STAR_DRAWS = comb(BONUS_NUMBER_RANGE, N_BONUS)    # 66
_CHUNK = 262_144

_DRAW_MASKS: Optional[np.ndarray] = None


def all_main_draw_masks() -> np.ndarray:
    """All C(50,5) main draws as uint64 bitmasks (cached after first call)."""
    global _DRAW_MASKS
    if _DRAW_MASKS is None:
        _DRAW_MASKS = np.fromiter(
            (sum(1 << (number - 1) for number in combo)
             for combo in itertools.combinations(range(1, MAIN_NUMBER_RANGE + 1), N_MAIN)),
            dtype=np.uint64,
            count=_TOTAL_MAIN_DRAWS,
        )
    return _DRAW_MASKS


_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def _popcount64_lut(values: np.ndarray) -> np.ndarray:
    """16-bit lookup-table popcount; fallback for NumPy < 2.0."""
    values = values.astype(np.uint64, copy=False)
    counts = _POPCOUNT_LUT[(values & np.uint64(0xFFFF)).astype(np.intp)].astype(np.uint8)
    for shift in (16, 32, 48):
        counts += _POPCOUNT_LUT[((values >> np.uint64(shift)) & np.uint64(0xFFFF)).astype(np.intp)]
    return counts


if hasattr(np, "bitwise_count"):
    def popcount64(values: np.ndarray) -> np.ndarray:
        return np.bitwise_count(values)
else:  # pragma: no cover - exercised only on NumPy < 2.0
    popcount64 = _popcount64_lut


def _ticket_mask(mains: Sequence[int]) -> np.uint64:
    return np.uint64(sum(1 << (int(number) - 1) for number in mains))


def _star_pair_index(stars: Sequence[int]) -> int:
    low, high = sorted(int(value) for value in stars)
    return (low - 1) * BONUS_NUMBER_RANGE + (high - 1)  # unique id per unordered pair


def single_ticket_prize_probability() -> float:
    """Exact P(one ticket wins any prize): >=2 mains, or exactly 1 main + both stars."""
    p_two_plus = sum(
        comb(N_MAIN, k) * comb(MAIN_NUMBER_RANGE - N_MAIN, N_MAIN - k)
        for k in range(2, N_MAIN + 1)
    ) / _TOTAL_MAIN_DRAWS
    p_one = comb(N_MAIN, 1) * comb(MAIN_NUMBER_RANGE - N_MAIN, N_MAIN - 1) / _TOTAL_MAIN_DRAWS
    return p_two_plus + p_one / _TOTAL_STAR_DRAWS


@dataclass
class PortfolioStats:
    num_tickets: int
    p_any_prize: float
    p_two_plus_mains: float
    p_three_plus_mains: float
    p_four_plus_mains: float
    max_main_match_distribution: Dict[int, float]
    expected_winning_tickets: float
    portfolio_ev_eur: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "num_tickets": self.num_tickets,
            "p_any_prize": round(self.p_any_prize, 6),
            "p_two_plus_mains": round(self.p_two_plus_mains, 6),
            "p_three_plus_mains": round(self.p_three_plus_mains, 6),
            "p_four_plus_mains": round(self.p_four_plus_mains, 8),
            "max_main_match_distribution": {
                str(k): round(v, 6) for k, v in self.max_main_match_distribution.items()
            },
            "expected_winning_tickets": round(self.expected_winning_tickets, 6),
        }
        if self.portfolio_ev_eur is not None:
            payload["portfolio_ev_eur"] = round(self.portfolio_ev_eur, 4)
        payload.update(self.extras)
        return payload


def evaluate_portfolio(tickets: Sequence[Ticket]) -> PortfolioStats:
    """
    Exact win structure of a set of tickets, by full enumeration of all
    2,118,760 main draws with analytic star folding. Exact, not simulated.
    """
    if not tickets:
        raise ValueError("Portfolio is empty.")
    for mains, stars in tickets:
        if len(set(mains)) != N_MAIN or not all(1 <= n <= MAIN_NUMBER_RANGE for n in mains):
            raise ValueError(f"Invalid main numbers: {mains}")
        if len(set(stars)) != N_BONUS or not all(1 <= n <= BONUS_NUMBER_RANGE for n in stars):
            raise ValueError(f"Invalid star numbers: {stars}")

    num_tickets = len(tickets)
    masks = np.array([_ticket_mask(mains) for mains, _ in tickets], dtype=np.uint64)

    # Map each ticket's star pair to a bit; equal pairs share a bit so an
    # OR + popcount counts *distinct* star pairs among m==1 tickets.
    pair_ids = [_star_pair_index(stars) for _, stars in tickets]
    unique_pairs = {pair: rank for rank, pair in enumerate(dict.fromkeys(pair_ids))}
    star_bits = np.array([np.uint64(1) << np.uint64(unique_pairs[p]) for p in pair_ids], dtype=np.uint64)

    draws = all_main_draw_masks()
    win_weight = 0.0                       # sum over draws of P(win | main draw)
    max_hist = np.zeros(N_MAIN + 1, dtype=np.int64)
    three_plus = 0
    four_plus = 0

    for start in range(0, len(draws), _CHUNK):
        chunk = draws[start : start + _CHUNK, None]           # (c, 1)
        matches = popcount64(chunk & masks[None, :])          # (c, T) match counts
        max_matches = matches.max(axis=1)
        max_hist += np.bincount(max_matches, minlength=N_MAIN + 1)
        three_plus += int((max_matches >= 3).sum())
        four_plus += int((max_matches >= 4).sum())

        certain = max_matches >= 2                            # tier 2+0 exists
        win_weight += float(certain.sum())

        maybe = ~certain & (matches == 1).any(axis=1)
        if maybe.any():
            star_or = np.bitwise_or.reduce(
                np.where(matches[maybe] == 1, star_bits[None, :], np.uint64(0)), axis=1
            )
            win_weight += float(popcount64(star_or).sum()) / _TOTAL_STAR_DRAWS

    total = float(_TOTAL_MAIN_DRAWS)
    p_single = single_ticket_prize_probability()
    return PortfolioStats(
        num_tickets=num_tickets,
        p_any_prize=win_weight / total,
        p_two_plus_mains=float(max_hist[2:].sum()) / total,
        p_three_plus_mains=three_plus / total,
        p_four_plus_mains=four_plus / total,
        max_main_match_distribution={k: float(max_hist[k]) / total for k in range(N_MAIN + 1)},
        expected_winning_tickets=num_tickets * p_single,
    )


# --- Construction ---------------------------------------------------------------

def _unpopular_star_pairs(ev_model: Any, count: int) -> List[Tuple[int, int]]:
    """Distinct star pairs, cheapest-to-share first. Distinctness maximises the
    1-main + 2-stars fallback coverage; unpopularity maximises EV."""
    weights = ev_model.bonus_weights
    pairs = sorted(
        itertools.combinations(range(1, BONUS_NUMBER_RANGE + 1), N_BONUS),
        key=lambda pair: weights[pair[0] - 1] * weights[pair[1] - 1],
    )
    return [tuple(pair) for pair in pairs[:count]]


def _ev_polish_disjoint(tickets: List[List[int]], ev_model: Any, stars: Sequence[Sequence[int]]) -> List[List[int]]:
    """Swap ticket numbers with unused numbers when it raises ticket EV.
    Main-side win structure is permutation-invariant for disjoint portfolios,
    so these swaps are free: pure EV gain, zero structural cost."""
    used = {number for ticket in tickets for number in ticket}
    unused = [n for n in range(1, MAIN_NUMBER_RANGE + 1) if n not in used]
    improved = True
    while improved and unused:
        improved = False
        for t_idx, ticket in enumerate(tickets):
            current_ev = ev_model.ticket_ev(ticket, stars[t_idx])["ev_eur"]
            for pos in range(N_MAIN):
                best_gain, best_swap = 0.0, None
                for u_idx, candidate in enumerate(unused):
                    trial = ticket.copy()
                    trial[pos] = candidate
                    gain = ev_model.ticket_ev(trial, stars[t_idx])["ev_eur"] - current_ev
                    if gain > best_gain + 1e-9:
                        best_gain, best_swap = gain, u_idx
                if best_swap is not None:
                    old = ticket[pos]
                    ticket[pos] = unused[best_swap]
                    unused[best_swap] = old
                    current_ev += best_gain
                    improved = True
            tickets[t_idx] = sorted(ticket)
    return [sorted(ticket) for ticket in tickets]


def build_portfolio(
    num_tickets: int,
    ev_model: Any,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Construct a portfolio with (a) exact-optimal main-side structure via
    disjointness while numbers last, (b) distinct unpopular star pairs,
    (c) EV-polished number choice. Returns candidate-record dicts with EV
    fields attached.
    """
    from .coverage import build_candidate_record

    if num_tickets <= 0:
        return []

    rng = np.random.default_rng(seed)
    inv = (1.0 / ev_model.main_weights)
    inv = inv / inv.sum()

    tickets: List[List[int]] = []
    seen_tickets: set = set()
    used: set = set()
    for _ in range(num_tickets):
        available = np.array([n for n in range(1, MAIN_NUMBER_RANGE + 1) if n not in used])
        if len(available) < N_MAIN:
            used = set()  # pool exhausted (T*5 > 50): start a fresh disjoint layer
            available = np.array([n for n in range(1, MAIN_NUMBER_RANGE + 1) if n not in used])
        probs = inv[available - 1]
        probs = probs / probs.sum()
        for _attempt in range(50):  # layered portfolios could re-draw an earlier line
            pick = sorted(int(n) for n in rng.choice(available, size=N_MAIN, replace=False, p=probs))
            if tuple(pick) not in seen_tickets:
                break
        tickets.append(pick)
        seen_tickets.add(tuple(pick))
        used.update(pick)

    star_pairs = _unpopular_star_pairs(ev_model, num_tickets)
    if len(star_pairs) < num_tickets:  # > 66 tickets: cycle
        star_pairs = list(itertools.islice(itertools.cycle(star_pairs), num_tickets))

    tickets = _ev_polish_disjoint(tickets, ev_model, star_pairs)

    uniform_main = np.full(MAIN_NUMBER_RANGE, 1.0 / MAIN_NUMBER_RANGE)
    uniform_bonus = np.full(BONUS_NUMBER_RANGE, 1.0 / BONUS_NUMBER_RANGE)
    records = [
        build_candidate_record(tuple(mains), tuple(stars), uniform_main, uniform_bonus, source="Portfolio")
        for mains, stars in zip(tickets, star_pairs)
    ]
    ev_model.attach(records)
    return records


def compare_to_random(
    tickets: Sequence[Ticket],
    ev_model: Any = None,
    n_baselines: int = 8,
    seed: int = 123,
) -> Dict[str, Any]:
    """Exact-evaluate random quick-pick portfolios of the same size, for an
    honest same-budget baseline."""
    rng = np.random.default_rng(seed)
    p_any, p_three, evs = [], [], []
    for _ in range(n_baselines):
        random_tickets = []
        for _ in range(len(tickets)):
            mains = tuple(sorted(rng.choice(np.arange(1, MAIN_NUMBER_RANGE + 1), size=N_MAIN, replace=False).tolist()))
            stars = tuple(sorted(rng.choice(np.arange(1, BONUS_NUMBER_RANGE + 1), size=N_BONUS, replace=False).tolist()))
            random_tickets.append((mains, stars))
        stats = evaluate_portfolio(random_tickets)
        p_any.append(stats.p_any_prize)
        p_three.append(stats.p_three_plus_mains)
        if ev_model is not None:
            evs.append(sum(ev_model.ticket_ev(m, s)["ev_eur"] for m, s in random_tickets))
    result = {
        "n_baselines": n_baselines,
        "mean_p_any_prize": float(np.mean(p_any)),
        "mean_p_three_plus_mains": float(np.mean(p_three)),
    }
    if evs:
        result["mean_portfolio_ev_eur"] = float(np.mean(evs))
    return result

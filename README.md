# EuroMillions Ticket Toolkit

An honest toolkit for a EuroMillions-style draw (5/50 mains + 2/12 stars).
It does **not** predict numbers — nothing can — but it squeezes out every
legitimate edge a fair lottery leaves a player, with exact math and
empirically calibrated inputs.

## What is and is not possible

Draws come from audited physical machines: balls matched to ~0.1 mg,
machines rotated and independently tested. The process has no memory and no
learnable state, so **every ticket has identical odds in every tier**.
Frequency analysis, hot/cold numbers, LSTMs, gradient boosting — all of it
is noise-fitting. Earlier versions of this project implemented those models
and measured their walk-forward lift: zero, exactly as theory predicts. They
have been removed.

What ticket choice *does* control — because most prize tiers are
pari-mutuel (split among winners) — is **how many people you share with
when you do win**. Humans over-pick birthdays, lucky numbers, and visual
patterns. Four real levers remain, and this toolkit implements all of
them:

1. **EV steering** — pick combinations fewer people play (~+20% expected
   payout per ticket).
2. **Jackpot timing** — EV per euro varies ~3.5x between a EUR 17M and a
   EUR 250M jackpot, even after modelling the sales surge on big draws.
3. **Portfolio structure** — for multi-ticket play, move probability mass
   toward "at least one ticket wins" (exactly computed, ~+10 points at 10
   tickets vs random quick-picks).
4. **Rule-mechanics events** — the capped **must-be-won draw** (5th draw at
   the EUR 250M cap): if nobody matches 5+2, the whole jackpot rolls down to
   the next winning tier. Because EUR 250M is then paid out with certainty
   over ~62M tickets, the modelled EV is ~EUR 4.45 per EUR 2.50 ticket —
   the *only* EV-positive draw type in the game. Rolldowns have really
   happened (17 Nov 2006: EUR 183M split by twenty 5+1 winners), and this is
   the same mechanism the MIT syndicate exploited in Massachusetts' Cash
   WinFall. Price one with `--must-be-won`.

On every ordinary draw, EV stays below the EUR 2.50 ticket price no matter
what you do — anyone claiming otherwise is selling something. The
must-be-won exception is created by the rules, not by prediction, is rare
(most cap cycles end in a jackpot win first), and being EV-positive still
means a ~1-in-7-million event carries the mean: you almost surely lose that
EUR 2.50. Unpopular lines amplify the rolldown further (fewer co-winners in
the receiving tier: ~EUR 8/ticket modelled at 0.3x crowding).

## Quick start from scratch

Requires Python 3.9+.

```bash
git clone <this-repo> && cd lotery_analysis   # or copy the folder
python3 -m venv .venv
source .venv/bin/activate                      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Confirm everything works (fast, ~5s)
python3 -m unittest discover -s tests -v
```

### The one command

Look up the advertised jackpot for the next draw, decide your budget, run:

```bash
python3 main.py --jackpot 130000000 --budget 12.50
```

That prints a complete **play plan**: a verdict on whether this draw is
worth playing at all, the exact lines to put on the playslip, why those
lines (co-winner crowding, EV in euros), and the exact odds of the set vs
the same budget on random quick-picks. Also saved to
`predictions/play_plan.json`. With no flags at all, `python3 main.py`
assumes the EUR 17M minimum jackpot and 5 tickets. Same `--seed` = same
lines; change it for a fresh set.

### Supporting commands

```bash
python3 main.py --ev-table      # EV per ticket across jackpot levels (when to play)
python3 main.py --validate-ev   # check the co-winner model against real winner counts
```

The repo ships with calibrated inputs (`lottery_data/winner_counts.csv`,
`lottery_data/popularity_fitted.json`), so everything above works
immediately. Results are also written as JSON to `predictions/`.

### Keeping the data fresh (recommended twice a year)

The only data that matters is **what other players pick**, and its public
signal is the official FDJ archives recording the Europe-wide number of
winners at each of the 13 prize ranks per draw. To update:

1. Download the *latest* EuroMillions history ZIP from
   [fdj.fr](https://www.fdj.fr/jeux-de-tirage/euromillions-my-million/historique)
   and drop it into `lottery_data/` (no need to unzip, no need to keep old
   ZIPs — new archives are merged into the cached history, never replace it).
2. `python3 main.py --calibrate-popularity`

That refreshes both the popularity weights and the sales estimate; every
command picks them up automatically, and the play plan will nudge you when
the calibration is more than ~6 months old.

**Note on `lottery_data/lottery_numbers.txt`:** this file of past drawn
numbers is inert — nothing reads it anymore, by design. Drawn numbers carry
zero information about future draws, so appending new results there has no
effect on anything. Keep it as a personal record or delete it; the pipeline
only consumes the winner-count archives.

## How each piece works

**EV engine (`src/ev.py`).** Prices every ticket in euros across all 13
tiers: exact hypergeometric tier probabilities x official fund allocations
(EUR 1.10 per line into the Common Prize Fund; jackpot tier uses the
advertised jackpot) x expected pari-mutuel share `E[1/(1+K)]` with
`K ~ Poisson(lambda)`. The co-winner intensity comes from the crowding
model calibrated on the FDJ winner counts. Full-main-match tiers price the
exact line (product of pick weights, elementary-symmetric normalised, times
documented pattern priors), so 1-2-3-4-5 is correctly priced as
catastrophically shared (~200x crowding). `--validate-ev` checks the model
against reality: on 1,013 archived draws it explains 26-60% of per-tier
winner-share variance (uniform baseline ~0, mean R^2 improvement +0.40).

**Exact portfolio structure (`src/portfolio.py`).** Enumerates all
C(50,5) = 2,118,760 possible main draws (vectorised popcount, ~0.2s) and
folds stars in analytically — exact probabilities, no simulation.
Construction uses disjoint mains (a permutation-invariance argument makes
any disjoint portfolio exactly equivalent on the main side, so number choice
is spent purely on EV), distinct unpopular star pairs, then an EV polish.
The printout always shows `E[winning tickets]`, which is identical for
every portfolio — structure only reshapes the distribution, never the mean.
Guarantee-style wheels (La Jolla Covering Repository; Cushing & Stewart's
27-ticket UK Lotto construction) are the classical alternative; famously,
the guaranteed win usually pays less than the tickets cost, which is why
this repo reports exact probabilities and euros instead.

**Popularity calibration (`src/popularity_fit.py`, `src/winner_data.py`).**
Within-draw ratios of adjacent prize tiers cancel ticket sales and every
draw-level effect, leaving a direct per-draw measurement of how over-picked
the drawn numbers were (the Riedwyl & Henze / Farrell et al. approach).
Weighted least squares on interpretable features recovers per-number pick
weights; holdout metrics are reported next to the heuristic priors they
replace. Current fit: calendar numbers ~+10%, number 7 +23%, stars 10-12
about -20% (legacy playslip habits).

**Ticket selection (`src/coverage.py`).** Builds candidate pools (including
anti-popular candidates sampled inversely to pick weights), then selects a
diversified set scored by EV with overlap penalties and coverage bonuses.

## Project layout

```
main.py                  CLI: play plan by default; --ev-table / --validate-ev / --calibrate-popularity
src/config.py            game constants, paths, selection defaults
src/ev.py                economic EV engine + crowding validation
src/portfolio.py         exact portfolio evaluation + construction
src/coverage.py          candidate pools + diversified selection
src/popularity.py        pick-weight model (fitted file with heuristic fallback)
src/popularity_fit.py    calibration from FDJ winner counts
src/winner_data.py       FDJ archive ingestion (ZIP/CSV, defensive parsing)
tests/                   50 tests incl. exactness proofs vs Monte Carlo
lottery_data/            calibrated inputs (+ drop FDJ ZIPs here to refit)
predictions/             JSON outputs (gitignored)
```

## The complete honest playbook

Skip base draws. If you play, play big rollovers — and if a cap cycle ever
reaches its 5th, must-be-won draw, that is the single best ticket the game
sells (`python3 main.py --jackpot 250000000 --must-be-won --budget <EUR>`).
Always use unpopular lines and disjoint sets; add your country's raffle via
`--raffle-prize/--raffle-pool` (raffle odds are better on Tuesdays). And
remember what the EV engine keeps printing: in expectation you lose on
every ordinary draw, and even an EV-positive must-be-won ticket almost
surely loses its EUR 2.50. Entertainment, not investment.

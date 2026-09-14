"""The measurement harness: per-category ASR with intervals, and the ablation."""

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence

from .corpus import Attack
from .stats import (
    Interval,
    minimum_detectable_difference,
    paired_monotone_delta,
    wilson,
)
from .target import Target, stack_latency_ms
from .taxonomy import CATEGORIES, DEFENCES, DEFENCE_LATENCY_MS


@dataclass(frozen=True)
class RunResult:
    """Outcome of one defence configuration over the whole corpus."""

    enabled: FrozenSet[str]
    outcomes: Dict[str, bool]          # attack_id -> attack succeeded
    by_category: Dict[str, List[Attack]]

    def successes(self, category: Optional[str] = None) -> int:
        return sum(1 for a in self._attacks(category) if self.outcomes[a.attack_id])

    def n(self, category: Optional[str] = None) -> int:
        return len(self._attacks(category))

    def asr(self, category: Optional[str] = None) -> Interval:
        return wilson(self.successes(category), self.n(category))

    def _attacks(self, category: Optional[str]) -> List[Attack]:
        if category is None:
            return [a for cat in self.by_category for a in self.by_category[cat]]
        return self.by_category[category]

    @property
    def latency_ms(self) -> float:
        return stack_latency_ms(self.enabled)


def run(target: Target, corpus: Sequence[Attack],
        enabled: Iterable[str]) -> RunResult:
    enabled = frozenset(enabled)
    by_category: Dict[str, List[Attack]] = {}
    outcomes: Dict[str, bool] = {}
    for attack in corpus:
        by_category.setdefault(attack.category, []).append(attack)
        outcomes[attack.attack_id] = bool(target.attempt(attack, enabled))
    return RunResult(enabled=enabled, outcomes=outcomes, by_category=by_category)


@dataclass(frozen=True)
class AblationRow:
    """One ablation arm: the full stack minus `removed`."""

    removed: FrozenSet[str]
    asr_full: float
    asr_ablated: float
    delta: float
    delta_ci: Interval
    latency_saved_ms: float

    @property
    def contributes(self) -> bool:
        """Does removing this set measurably raise ASR?

        The test is whether the interval on the delta excludes zero - i.e.
        whether at least one trial flipped. With a paired monotone design a
        single flipped trial is real evidence, whereas zero flips across the
        whole corpus is the signature of a layer whose coverage is entirely
        contained in another layer's.
        """
        return self.delta_ci.successes > 0

    @property
    def ms_per_point(self) -> Optional[float]:
        """Milliseconds of latency bought per percentage point of ASR removed."""
        if self.delta <= 0:
            return None
        return self.latency_saved_ms / (100.0 * self.delta)


def ablate(target: Target, corpus: Sequence[Attack],
           full_stack: Iterable[str] = DEFENCES,
           max_combination: int = 2) -> List[AblationRow]:
    """Measure ASR with each defence removed in turn, and in combination.

    Single removals alone are not enough. Two layers that cover the same
    attacks will EACH look useless when removed alone, because the other one
    catches everything; only the joint removal reveals that the coverage was
    real but duplicated. Conversely a layer that looks useless alone and still
    contributes nothing when removed alongside its suspected twin is genuinely
    dead weight. Reporting only single-layer deltas is the standard mistake.
    """
    full_stack = frozenset(full_stack)
    baseline = run(target, corpus, full_stack)
    n = baseline.n()
    asr_full = baseline.successes() / n

    rows: List[AblationRow] = []
    for size in range(1, max_combination + 1):
        for combo in combinations(sorted(full_stack), size):
            removed = frozenset(combo)
            arm = run(target, corpus, full_stack - removed)
            gained = lost = 0
            for attack_id, succeeded in arm.outcomes.items():
                was = baseline.outcomes[attack_id]
                if succeeded and not was:
                    gained += 1
                elif was and not succeeded:
                    lost += 1
            delta, ci = paired_monotone_delta(gained, lost, n)
            rows.append(AblationRow(
                removed=removed,
                asr_full=asr_full,
                asr_ablated=arm.successes() / n,
                delta=delta,
                delta_ci=ci,
                latency_saved_ms=sum(DEFENCE_LATENCY_MS[d] for d in removed),
            ))
    rows.sort(key=lambda r: (len(r.removed), -r.delta))
    return rows


def redundant_defences(rows: Sequence[AblationRow]) -> List[str]:
    """Layers with zero measured contribution when removed on their own.

    Named "redundant" and not "useless" on purpose: a layer here may well block
    attacks, just none that some other enabled layer does not already block.
    Drop it and ASR does not move; keep it and every request pays its latency.
    """
    out = []
    for row in rows:
        if len(row.removed) == 1 and not row.contributes:
            out.extend(row.removed)
    return sorted(out)


def masked_by(rows: Sequence[AblationRow], defence: str) -> List[str]:
    """Layers whose joint removal with `defence` unlocks more than either alone.

    This is what distinguishes "redundant given the stack" from "does nothing
    at all", and it is the answer to 'which guardrail would you drop'.
    """
    singles = {next(iter(r.removed)): r.delta
               for r in rows if len(r.removed) == 1}
    out = []
    for row in rows:
        if len(row.removed) == 2 and defence in row.removed:
            other = next(iter(row.removed - {defence}))
            if row.delta > singles[defence] + singles[other] + 1e-12:
                out.append(other)
    return sorted(out)


@dataclass(frozen=True)
class Report:
    baseline: RunResult
    undefended: RunResult
    per_category: Dict[str, Interval]
    aggregate: Interval
    worst_category: str
    ablation: List[AblationRow]
    mdd_per_category: Dict[str, float]

    @property
    def headline_block_rate(self) -> float:
        """The number a generic harness would print, and nothing else."""
        return 1.0 - self.aggregate.point

    @property
    def aggregate_understates_worst_by(self) -> float:
        return self.per_category[self.worst_category].point - self.aggregate.point


def evaluate(target: Target, corpus: Sequence[Attack],
             full_stack: Iterable[str] = DEFENCES,
             max_combination: int = 2) -> Report:
    full_stack = frozenset(full_stack)
    baseline = run(target, corpus, full_stack)
    undefended = run(target, corpus, frozenset())
    cats = [c for c in CATEGORIES if c in baseline.by_category]
    per_category = {c: baseline.asr(c) for c in cats}
    worst = max(cats, key=lambda c: per_category[c].point)
    mdd = {c: minimum_detectable_difference(baseline.n(c), per_category[c].point)
           for c in cats}
    return Report(
        baseline=baseline,
        undefended=undefended,
        per_category=per_category,
        aggregate=baseline.asr(),
        worst_category=worst,
        ablation=ablate(target, corpus, full_stack, max_combination),
        mdd_per_category=mdd,
    )


def regression_check(before: Report, after: Report) -> List[str]:
    """CI gate: which categories got measurably worse between two runs?

    Compares per-category Wilson intervals and flags only NON-OVERLAPPING
    increases. A bare point-estimate comparison would fire on sampling noise
    every other commit and be switched off within a week; requiring separation
    of the intervals means a flag is a claim the trial count can support.

    Deliberately per-category and not on the aggregate: a model change that
    doubles roleplay-framing ASR while shaving a point off three large, easy
    categories leaves the aggregate flat. That is precisely the regression you
    most need to catch.
    """
    flagged = []
    for category, after_ci in after.per_category.items():
        before_ci = before.per_category.get(category)
        if before_ci is None:
            continue
        if after_ci.low > before_ci.high:
            flagged.append(category)
    return sorted(flagged)

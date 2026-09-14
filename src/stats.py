"""Binomial interval estimation for small, lopsided samples.

Everything here is pure stdlib `math`. No scipy.
"""

import math
from dataclasses import dataclass
from typing import Tuple

# Two-sided 95% normal quantile, and the 80%-power one-sided quantile. Both are
# the standard conventions, not tuned numbers: alpha=0.05, power=0.80.
Z_95 = 1.959963984540054
Z_POWER_80 = 0.8416212335729143


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Present so callers can request a non-default alpha without scipy. Accurate
    to about 1.15e-9 in the relative sense, far beyond what any interval
    reported here needs.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def z_for(alpha: float) -> float:
    # Short-circuit the default so the overwhelmingly common 95% interval uses
    # the exact quantile rather than an approximation good to only ~1e-9.
    if alpha == 0.05:
        return Z_95
    return _norm_ppf(1.0 - alpha / 2.0)


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    n: int
    successes: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def fmt(self) -> str:
        return "%6.1f%% [%5.1f%%, %5.1f%%]" % (
            100 * self.point, 100 * self.low, 100 * self.high)


def wilson(successes: int, n: int, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a binomial proportion.

    Wilson and not Wald. Attack success rates are exactly the regime where Wald
    falls apart: rates near 0 or 1, and per-category trial counts in the tens.
    At k=0 Wald returns the degenerate interval [0, 0] - a claim of certainty
    from twenty trials - while Wilson returns a positive upper bound that
    honestly reflects what twenty trials can rule out. `wald()` below exists
    so a test can demonstrate the failure rather than assert it in prose.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes out of range")
    z = z_for(alpha)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    low, high = centre - half, centre + half
    # The bounds are exactly 0 at k=0 and exactly 1 at k=n; cancellation in the
    # subtraction leaves ~1e-19 residue instead. Snap them, because downstream
    # code tests these edges for equality.
    if successes == 0:
        low = 0.0
    if successes == n:
        high = 1.0
    return Interval(point=p, low=max(0.0, low), high=min(1.0, high),
                    n=n, successes=successes)


def wald(successes: int, n: int, alpha: float = 0.05) -> Interval:
    """Textbook Wald interval. Kept only as the foil the tests compare against."""
    z = z_for(alpha)
    p = successes / n
    half = z * math.sqrt(p * (1 - p) / n)
    return Interval(point=p, low=max(0.0, p - half), high=min(1.0, p + half),
                    n=n, successes=successes)


def minimum_detectable_difference(n_per_arm: int, baseline_rate: float,
                                  alpha: float = 0.05,
                                  power: float = 0.80) -> float:
    """Smallest absolute difference in rates two arms of `n_per_arm` could resolve.

    Standard two-proportion normal approximation:
        delta = (z_{alpha/2} + z_{power}) * sqrt(2 * p * (1-p) / n)

    Reported so a reader knows what the benchmark could NOT have resolved. A
    defence whose true effect is below this line will show up as "no measured
    contribution" no matter how many times the suite is run - that is a
    statement about the experiment, not about the defence, and conflating the
    two is how benchmarks lie.
    """
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    p = min(max(baseline_rate, 1e-9), 1 - 1e-9)
    z_a = z_for(alpha)
    z_b = _norm_ppf(power)
    return (z_a + z_b) * math.sqrt(2.0 * p * (1 - p) / n_per_arm)


def paired_monotone_delta(gained: int, lost: int, n: int,
                          alpha: float = 0.05) -> Tuple[float, Interval]:
    """Interval for the ASR increase caused by removing one defence.

    The ablation is PAIRED: the identical trials run in both arms. Removing a
    defence from a stack can only ever turn a block into a success, never the
    reverse, so one of the two discordant cells is structurally zero. Given
    that, the delta is just the proportion of trials in the single non-empty
    discordant cell and a Wilson interval on it is the right answer.

    `lost` is asserted rather than modelled: if it is ever non-zero the
    monotonicity assumption has broken (a non-monotone stack, e.g. a defence
    that rewrites input in a way that helps an attack) and the caller needs
    Newcombe's method or McNemar's test instead of this shortcut. Failing
    loudly beats reporting a number computed under an assumption that no
    longer holds.
    """
    if lost:
        raise ValueError(
            "non-monotone ablation (%d trials got SAFER without the defence); "
            "use Newcombe/McNemar, not paired_monotone_delta" % lost)
    return gained / n, wilson(gained, n, alpha)

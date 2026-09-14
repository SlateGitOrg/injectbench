"""Interval estimation: Wilson vs the Wald interval a generic harness reaches for."""

import math
import random
import unittest

from src.stats import (
    Z_95,
    _norm_ppf,
    minimum_detectable_difference,
    paired_monotone_delta,
    wald,
    wilson,
)


class TestNormalQuantile(unittest.TestCase):
    def test_matches_published_quantiles(self):
        # Standard normal quantiles to 6dp. If the rational approximation is
        # wrong every interval in the repo is wrong, so pin it hard.
        for p, expected in ((0.975, 1.959964), (0.95, 1.644854),
                            (0.80, 0.841621), (0.5, 0.0), (0.01, -2.326348)):
            self.assertAlmostEqual(_norm_ppf(p), expected, places=6)

    def test_symmetric(self):
        for p in (0.001, 0.02, 0.3, 0.49):
            self.assertAlmostEqual(_norm_ppf(p), -_norm_ppf(1 - p), places=9)


class TestWilsonClosedForm(unittest.TestCase):
    def test_against_hand_computed_value(self):
        # Wilson recomputed here from the textbook formula independently of the
        # implementation's factoring, so an algebra slip cannot hide.
        k, n, z = 7, 40, Z_95
        p = k / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        ci = wilson(k, n)
        self.assertAlmostEqual(ci.low, centre - half, places=12)
        self.assertAlmostEqual(ci.high, centre + half, places=12)

    def test_zero_successes_gives_positive_upper_bound(self):
        ci = wilson(0, 20)
        self.assertEqual(ci.low, 0.0)
        self.assertGreater(ci.high, 0.15)
        # ... and the naive alternative claims certainty from 20 trials.
        self.assertEqual(wald(0, 20).high, 0.0)

    def test_all_successes_stays_inside_unit_interval(self):
        ci = wilson(30, 30)
        self.assertLessEqual(ci.high, 1.0)
        self.assertLess(ci.low, 1.0)


class TestWilsonBeatsWald(unittest.TestCase):
    """The differentiator's statistical half, measured rather than asserted."""

    def test_coverage_at_low_rate_small_n(self):
        # p=0.03, n=40 is exactly the injection-benchmark regime: a rare event
        # and a modest per-category trial count. Nominal coverage is 95%.
        p_true, n, reps = 0.03, 40, 4000
        rng = random.Random(11)
        wilson_hits = wald_hits = 0
        for _ in range(reps):
            k = sum(1 for _ in range(n) if rng.random() < p_true)
            wilson_hits += wilson(k, n).contains(p_true)
            wald_hits += wald(k, n).contains(p_true)
        wilson_cov = wilson_hits / reps
        wald_cov = wald_hits / reps

        # Threshold derived, not chosen: the two estimators are evaluated on
        # the SAME draws, so compare with the paired standard error of the
        # coverage difference and require separation of 3 SE.
        diff = wilson_cov - wald_cov
        se = math.sqrt(diff * (1 - abs(diff)) / reps) if diff else 1.0
        self.assertGreater(diff, 3 * se,
                           "Wilson must beat Wald by more than sampling noise")
        # Wilson is near nominal; Wald is not even close at this rate.
        self.assertGreater(wilson_cov, 0.93)
        self.assertLess(wald_cov, 0.90)


class TestMinimumDetectableDifference(unittest.TestCase):
    def test_matches_two_proportion_formula(self):
        n, p = 400, 0.10
        expected = (1.959963984540054 + 0.8416212335729143) * \
            math.sqrt(2 * p * (1 - p) / n)
        self.assertAlmostEqual(minimum_detectable_difference(n, p), expected,
                               places=10)

    def test_shrinks_as_sqrt_n(self):
        a = minimum_detectable_difference(100, 0.2)
        b = minimum_detectable_difference(400, 0.2)
        self.assertAlmostEqual(a / b, 2.0, places=6)

    def test_twenty_trials_cannot_resolve_a_small_effect(self):
        # The claim the README makes about small categories, as a test.
        self.assertGreater(minimum_detectable_difference(20, 0.10), 0.26)


class TestPairedMonotoneDelta(unittest.TestCase):
    def test_delta_and_interval(self):
        delta, ci = paired_monotone_delta(gained=13, lost=0, n=500)
        self.assertAlmostEqual(delta, 0.026)
        self.assertTrue(ci.contains(0.026))
        self.assertGreater(ci.low, 0.0)

    def test_zero_gain_interval_touches_zero_but_bounds_the_effect(self):
        _, ci = paired_monotone_delta(0, 0, 2000)
        self.assertEqual(ci.low, 0.0)
        self.assertGreater(ci.high, 0.0)
        self.assertLess(ci.high, 0.003)

    def test_refuses_to_compute_when_monotonicity_breaks(self):
        # Silently absorbing a non-monotone result would report a number
        # computed under an assumption that no longer holds.
        with self.assertRaises(ValueError):
            paired_monotone_delta(gained=5, lost=2, n=100)


if __name__ == "__main__":
    unittest.main()

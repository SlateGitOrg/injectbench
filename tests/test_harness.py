"""The differentiator: per-taxonomy intervals, and an ablation that finds the
redundant layer a single aggregate number would never surface."""

import unittest

from src.corpus import generate_corpus
from src.harness import (
    evaluate,
    masked_by,
    redundant_defences,
    regression_check,
    run,
)
from src.stats import minimum_detectable_difference, wald, wilson
from src.target import SimulatedTarget, UndefendedControl
from src.taxonomy import (
    CATEGORIES,
    DEFENCES,
    DELIMITER_ISOLATION,
    DIRECT_OVERRIDE,
    INPUT_FILTER,
    PROFILES,
    PROVENANCE_TAGGER,
    ROLEPLAY_FRAMING,
)

N = 400
FULL = frozenset(DEFENCES)


class HarnessCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = generate_corpus(N)
        cls.target = SimulatedTarget()
        cls.report = evaluate(cls.target, cls.corpus)
        cls.singles = {next(iter(r.removed)): r
                       for r in cls.report.ablation if len(r.removed) == 1}


class TestPerCategoryReporting(HarnessCase):
    def test_intervals_cover_the_planted_truth_for_every_category(self):
        for category, ci in self.report.per_category.items():
            truth = PROFILES[category].residual_asr(FULL)
            self.assertTrue(ci.contains(truth),
                            "%s: truth %.4f outside %s" % (category, truth, ci.fmt()))

    def test_intervals_widen_when_trials_are_scarce(self):
        thin = evaluate(self.target, generate_corpus(20))
        for category in CATEGORIES:
            self.assertGreater(thin.per_category[category].width,
                               self.report.per_category[category].width)

    def test_minimum_detectable_difference_is_reported_and_honest(self):
        for category, mdd in self.report.mdd_per_category.items():
            self.assertGreater(mdd, 0.0)
            expected = minimum_detectable_difference(
                N, self.report.per_category[category].point)
            self.assertAlmostEqual(mdd, expected, places=12)
        # At 20 trials the floor is coarse enough that most real defence
        # effects would be invisible; the report has to say so rather than
        # print "no measured effect" and let the reader infer "no effect".
        thin = evaluate(self.target, generate_corpus(20))
        self.assertGreater(max(thin.mdd_per_category.values()), 0.25)


class TestAggregateIsMisleading(HarnessCase):
    """The headline number a generic harness prints, shown to be false comfort."""

    def test_headline_block_rate_looks_reassuring(self):
        self.assertGreater(self.report.headline_block_rate, 0.85)

    def test_but_the_worst_category_is_far_outside_the_aggregate_interval(self):
        worst = self.report.per_category[self.report.worst_category]
        self.assertEqual(self.report.worst_category, ROLEPLAY_FRAMING)
        # Not "a bit above average": the aggregate interval does not come near
        # containing it, so quoting the aggregate is a false statement about
        # this category rather than merely an imprecise one.
        self.assertFalse(self.report.aggregate.contains(worst.point))
        self.assertGreater(worst.low, self.report.aggregate.high)
        self.assertGreater(worst.point / self.report.aggregate.point, 3.0)

    def test_a_stack_that_looks_strong_overall_barely_moves_one_category(self):
        undef = self.report.undefended.asr(ROLEPLAY_FRAMING).point
        defended = self.report.per_category[ROLEPLAY_FRAMING].point
        overall_reduction = 1 - self.report.aggregate.point / \
            self.report.undefended.asr().point
        category_reduction = 1 - defended / undef
        self.assertGreater(overall_reduction, 0.85)
        self.assertLess(category_reduction, 0.5)

    def test_wald_on_a_small_category_would_assert_a_false_certainty(self):
        # The generic combination -- small per-category n plus a Wald interval
        # -- yields a zero-width interval that excludes the true rate.
        # Rates this low mean a 20-trial slice usually observes zero successes,
        # so the failure is the normal case and not a cherry-picked draw.
        zero_observed = []
        for category in CATEGORIES:
            res = run(self.target, generate_corpus(20, categories=(category,)),
                      FULL)
            k, n = res.successes(), res.n()
            truth = PROFILES[category].residual_asr(FULL)
            self.assertTrue(wilson(k, n).contains(truth), category)
            if k == 0:
                zero_observed.append(category)
                self.assertEqual(wald(k, n).width, 0.0)
                self.assertFalse(wald(k, n).contains(truth), category)
        self.assertGreaterEqual(len(zero_observed), 3)


class TestAblationFindsTheRedundantLayer(HarnessCase):
    """Without this the ablation is decoration."""

    def test_exactly_the_planted_redundant_layer_is_flagged(self):
        self.assertEqual(redundant_defences(self.report.ablation),
                         [PROVENANCE_TAGGER])

    def test_removing_it_flips_literally_no_trials(self):
        row = self.singles[PROVENANCE_TAGGER]
        self.assertEqual(row.delta_ci.successes, 0)
        self.assertEqual(row.delta, 0.0)
        self.assertEqual(row.asr_ablated, row.asr_full)
        self.assertIsNone(row.ms_per_point)

    def test_the_finding_is_correct_and_not_merely_underpowered(self):
        # Closed form agrees: stack ASR is identical with and without the
        # layer, so a measured delta of zero is the right answer rather than a
        # failure to detect a small one.
        for category in CATEGORIES:
            with_it = PROFILES[category].residual_asr(FULL)
            without = PROFILES[category].residual_asr(FULL - {PROVENANCE_TAGGER})
            self.assertAlmostEqual(with_it, without, places=12)

    def test_every_other_layer_does_contribute(self):
        for name, row in self.singles.items():
            if name == PROVENANCE_TAGGER:
                continue
            self.assertTrue(row.contributes, name)
            self.assertGreater(row.delta_ci.low, 0.0, name)

    def test_joint_removal_shows_masking_rather_than_nullity(self):
        # Single-layer ablation alone cannot tell "does nothing" from "does
        # something another layer already does". The joint arm can.
        joint = next(r for r in self.report.ablation
                     if r.removed == frozenset({PROVENANCE_TAGGER,
                                                DELIMITER_ISOLATION}))
        solo_delim = self.singles[DELIMITER_ISOLATION].delta
        solo_prov = self.singles[PROVENANCE_TAGGER].delta
        self.assertGreater(joint.delta, solo_delim + solo_prov)
        self.assertEqual(masked_by(self.report.ablation, PROVENANCE_TAGGER),
                         [DELIMITER_ISOLATION])

    def test_dropping_the_redundant_layer_has_a_bounded_cost(self):
        row = self.singles[PROVENANCE_TAGGER]
        self.assertGreater(row.latency_saved_ms, 0)
        # Bound what is given up by the interval, not by the point estimate of
        # zero: "we measured no effect" is a claim about an upper bound.
        self.assertLess(row.delta_ci.high, 0.005)

    def test_ablation_arms_are_paired_not_resampled(self):
        # Removing a defence must never make a trial safer. If the arms were
        # independently sampled this would fail, and the monotone delta
        # interval would be invalid.
        base = run(self.target, self.corpus, FULL).outcomes
        for defence in DEFENCES:
            arm = run(self.target, self.corpus, FULL - {defence}).outcomes
            for attack_id, was in base.items():
                if was:
                    self.assertTrue(arm[attack_id])


class TestControlsAndRegressionMode(HarnessCase):
    def test_harness_has_power_against_an_undefended_control(self):
        rep = evaluate(UndefendedControl(), self.corpus, max_combination=1)
        self.assertEqual(rep.aggregate.point, 1.0)
        # Every layer correctly reports zero contribution against a target that
        # ignores them all, so "zero delta" is not an artefact of the ablation.
        self.assertEqual(sorted(redundant_defences(rep.ablation)),
                         sorted(DEFENCES))

    def test_regression_mode_flags_a_real_per_category_increase(self):
        after = evaluate(self.target, self.corpus, FULL - {INPUT_FILTER})
        self.assertIn(DIRECT_OVERRIDE, regression_check(self.report, after))

    def test_regression_mode_does_not_flag_an_identical_run(self):
        self.assertEqual(regression_check(self.report, self.report), [])

    def test_regression_mode_is_silent_when_the_redundant_layer_is_dropped(self):
        # The operational payoff: dropping the layer the ablation called
        # redundant must not trip the CI gate on any category.
        after = evaluate(self.target, self.corpus, FULL - {PROVENANCE_TAGGER})
        self.assertEqual(regression_check(self.report, after), [])


if __name__ == "__main__":
    unittest.main()

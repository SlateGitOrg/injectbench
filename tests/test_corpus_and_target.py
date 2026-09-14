"""The simulated backend: reproducibility, planted ground truth, and controls."""

import unittest

from src.corpus import generate_corpus
from src.harness import run
from src.stats import wilson
from src.target import (
    HardRefusalBaseline,
    SimulatedTarget,
    UndefendedControl,
    _blocks,
)
from src.taxonomy import (
    CATEGORIES,
    CUE_UNTRUSTED_CHANNEL,
    DEFENCES,
    DELIMITER_ISOLATION,
    PROFILES,
    PROVENANCE_TAGGER,
)

N = 600
FULL = frozenset(DEFENCES)


class TestCorpusReproducibility(unittest.TestCase):
    def test_identical_across_runs(self):
        self.assertEqual(generate_corpus(50), generate_corpus(50))

    def test_attack_n_is_stable_when_corpus_grows(self):
        # If attack 7 changed when the corpus grew from 50 to 500, no ablation
        # result would be comparable across runs with different budgets.
        small = {a.attack_id: a for a in generate_corpus(50)}
        large = {a.attack_id: a for a in generate_corpus(500)}
        for attack_id, attack in small.items():
            self.assertEqual(attack, large[attack_id])

    def test_attack_is_stable_when_other_categories_are_dropped(self):
        one = generate_corpus(30, categories=(CATEGORIES[3],))
        both = {a.attack_id: a for a in generate_corpus(30)}
        for attack in one:
            self.assertEqual(attack, both[attack.attack_id])

    def test_ids_unique_and_categories_tagged(self):
        corpus = generate_corpus(N)
        self.assertEqual(len({a.attack_id for a in corpus}), len(corpus))
        self.assertEqual({a.category for a in corpus}, set(CATEGORIES))

    def test_cue_frequencies_recover_planted_probabilities(self):
        corpus = generate_corpus(2000)
        for category in CATEGORIES:
            attacks = [a for a in corpus if a.category == category]
            for cue, p_true in PROFILES[category].cue_probs.items():
                k = sum(1 for a in attacks if a.has(cue))
                ci = wilson(k, len(attacks))
                self.assertTrue(ci.contains(p_true),
                                "%s/%s: planted %.3f not in %s"
                                % (category, cue, p_true, ci.fmt()))


class TestSimulatedTarget(unittest.TestCase):
    def setUp(self):
        self.corpus = generate_corpus(N)
        self.target = SimulatedTarget()

    def test_deterministic(self):
        a = run(self.target, self.corpus, FULL).outcomes
        b = run(SimulatedTarget(), generate_corpus(N), FULL).outcomes
        self.assertEqual(a, b)

    def test_adding_a_defence_never_helps_the_attacker(self):
        # Monotonicity is what licenses the paired-delta shortcut in stats.
        base = run(self.target, self.corpus, frozenset()).outcomes
        for defence in DEFENCES:
            armed = run(self.target, self.corpus, frozenset({defence})).outcomes
            for attack_id, succeeded in armed.items():
                if succeeded:
                    self.assertTrue(base[attack_id])

    def test_undefended_asr_recovers_planted_base_rate(self):
        res = run(self.target, self.corpus, frozenset())
        for category in CATEGORIES:
            ci = res.asr(category)
            self.assertTrue(ci.contains(PROFILES[category].base_success),
                            "%s: planted %.3f not in %s" % (
                                category, PROFILES[category].base_success,
                                ci.fmt()))

    def test_defended_asr_matches_independent_closed_form(self):
        # CategoryProfile.residual_asr multiplies out the survival probabilities
        # analytically; the target simulates trial by trial. They are written
        # separately on purpose, so agreement is evidence rather than tautology.
        res = run(self.target, self.corpus, FULL)
        for category in CATEGORIES:
            expected = PROFILES[category].residual_asr(FULL)
            self.assertTrue(res.asr(category).contains(expected),
                            "%s: closed form %.4f not in %s" % (
                                category, expected, res.asr(category).fmt()))

    def test_susceptibility_depends_on_category(self):
        res = run(self.target, self.corpus, FULL)
        rates = sorted(res.asr(c).point for c in CATEGORIES)
        # Non-overlapping intervals between best and worst: the categories are
        # genuinely different, not noise around one number.
        best = min(CATEGORIES, key=lambda c: res.asr(c).point)
        worst = max(CATEGORIES, key=lambda c: res.asr(c).point)
        self.assertGreater(res.asr(worst).low, res.asr(best).high)
        self.assertGreater(rates[-1], 0.3)


class TestPlantedRedundancy(unittest.TestCase):
    def test_provenance_blocks_a_strict_subset_of_delimiter_isolation(self):
        corpus = generate_corpus(N)
        prov = {a.attack_id for a in corpus if _blocks(PROVENANCE_TAGGER, a)}
        delim = {a.attack_id for a in corpus if _blocks(DELIMITER_ISOLATION, a)}
        self.assertTrue(prov)                 # it does block things
        self.assertTrue(prov < delim)         # but never anything new
        self.assertTrue(any(a.has(CUE_UNTRUSTED_CHANNEL) for a in corpus))


class TestControls(unittest.TestCase):
    """A harness reporting low ASR because it is weak looks exactly like one
    reporting low ASR because the system is safe. These separate the two."""

    def setUp(self):
        self.corpus = generate_corpus(N)

    def test_undefended_control_is_fully_compromised(self):
        res = run(UndefendedControl(), self.corpus, FULL)
        self.assertEqual(res.asr().point, 1.0)
        for category in CATEGORIES:
            self.assertEqual(res.asr(category).point, 1.0)

    def test_hard_refusal_baseline_is_exactly_zero(self):
        res = run(HardRefusalBaseline(), self.corpus, FULL)
        self.assertEqual(res.asr().point, 0.0)
        # And the interval still admits a rate the sample could not rule out.
        self.assertGreater(res.asr().high, 0.0)

    def test_rejects_unknown_defence_name(self):
        with self.assertRaises(KeyError):
            SimulatedTarget().attempt(self.corpus[0], {"vibes_filter"})


if __name__ == "__main__":
    unittest.main()

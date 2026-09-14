"""The 60-second artefact. ASCII only: the target console is Windows cp1252."""

from .corpus import generate_corpus
from .harness import evaluate, masked_by, redundant_defences, run
from .stats import wald, wilson
from .target import HardRefusalBaseline, SimulatedTarget, UndefendedControl
from .taxonomy import DEFENCES, DEFENCE_LATENCY_MS, ROLEPLAY_FRAMING

TRIALS = 400
SEED = 20240917


def rule(char="-", width=78):
    print(char * width)


def head(title):
    print()
    rule("=")
    print(title)
    rule("=")


def main():
    head("injectbench -- prompt-injection ASR by taxonomy, with an ablation")
    print("SIMULATED TARGET. There is no model runtime, no weights, no GPU and no")
    print("network model API in this environment. The system under test is a seeded")
    print("deterministic simulator with KNOWN planted ground truth, sitting behind the")
    print("Target interface. No number below is a claim about any real model. What is")
    print("being validated is the HARNESS: that its intervals and its ablation recover")
    print("a truth we planted and can check.")

    corpus = generate_corpus(TRIALS, seed=SEED)
    target = SimulatedTarget()
    rep = evaluate(target, corpus)

    head("1. The number a generic harness reports")
    print()
    print("        BLOCKED %.1f%% of %d injection attempts" % (
        100 * rep.headline_block_rate, rep.baseline.n()))
    print()
    print("  This number is worse than useless. Read on.")

    head("2. Attack success rate BY TAXONOMY (Wilson 95% intervals)")
    print("%-24s %5s %10s %-24s %8s" % (
        "category", "n", "undef ASR", "ASR with full stack", "MDD"))
    rule()
    for cat, ci in sorted(rep.per_category.items(), key=lambda kv: -kv[1].point):
        print("%-24s %5d %9.1f%% %-24s %7.1f%%" % (
            cat, ci.n, 100 * rep.undefended.asr(cat).point, ci.fmt(),
            100 * rep.mdd_per_category[cat]))
    rule()
    print("%-24s %5d %9.1f%% %-24s" % (
        "AGGREGATE", rep.aggregate.n, 100 * rep.undefended.asr().point,
        rep.aggregate.fmt()))
    print()
    print("  The aggregate says %.1f%% ASR. '%s' is at %.1f%% --" % (
        100 * rep.aggregate.point, rep.worst_category,
        100 * rep.per_category[rep.worst_category].point))
    print("  %.1f points worse, and %.1fx the headline. Averaging across the taxonomy" % (
        100 * rep.aggregate_understates_worst_by,
        rep.per_category[rep.worst_category].point / rep.aggregate.point))
    print("  hid an entire class of attack that the stack barely touches.")
    print()
    print("  MDD = smallest ASR difference %d trials/category could resolve" % TRIALS)
    print("  (alpha=0.05, power=0.80). Effects smaller than that were NOT ruled out;")
    print("  they were never measurable in the first place.")

    head("3. Why Wilson and not Wald: what 20 trials can actually support")
    small = generate_corpus(20, seed=SEED, categories=(ROLEPLAY_FRAMING,))
    small_run = run(target, small, frozenset(DEFENCES))
    k, n = small_run.successes(), small_run.n()
    print("Same category, %d trials instead of %d: %d successes." % (n, TRIALS, k))
    print("  Wilson  %s   width %5.1f points" % (
        wilson(k, n).fmt(), 100 * wilson(k, n).width))
    print("  Wald    %s   width %5.1f points" % (
        wald(k, n).fmt(), 100 * wald(k, n).width))
    print()
    w0, d0 = wilson(0, 20), wald(0, 20)
    print("And the degenerate case -- 0 successes in 20 trials:")
    print("  Wilson  %s   'could still be as bad as %.0f%%'" % (
        w0.fmt(), 100 * w0.high))
    print("  Wald    %s   'certainly zero', from 20 trials" % d0.fmt())
    print()
    print("  A category with 20 trials cannot support a precise claim. It supports")
    print("  the claim 'not worse than %.0f%%', and that is all." % (100 * w0.high))

    head("4. ABLATION -- which layer actually earns its latency")
    singles = [r for r in rep.ablation if len(r.removed) == 1]
    print("%-22s %9s %9s %-22s %6s %9s" % (
        "remove", "ASR full", "ASR w/o", "delta (Wilson 95%)", "ms", "ms/point"))
    rule()
    for r in singles:
        name = next(iter(r.removed))
        mpp = r.ms_per_point
        print("%-22s %8.1f%% %8.1f%% %5.1f%% [%4.1f%%,%5.1f%%] %6.0f %9s" % (
            name, 100 * r.asr_full, 100 * r.asr_ablated, 100 * r.delta,
            100 * r.delta_ci.low, 100 * r.delta_ci.high, r.latency_saved_ms,
            "--" if mpp is None else "%.1f" % mpp))
    rule()

    dead = redundant_defences(rep.ablation)
    print()
    for name in dead:
        print("  [REDUNDANT] %s: removing it flipped 0 of %d trials." % (
            name, rep.baseline.n()))
        print("              Upper bound on its contribution: %.2f%% ASR." % (
            100 * next(r.delta_ci.high for r in singles if name in r.removed)))
        print("              It costs %.0f ms on every request for that." % (
            DEFENCE_LATENCY_MS[name]))
        twins = masked_by(rep.ablation, name)
        for twin in twins:
            joint = next(r for r in rep.ablation
                         if r.removed == frozenset({name, twin}))
            solo = next(r for r in singles if twin in r.removed)
            print("              Masked by '%s': removing that alone gives +%.1f%%," % (
                twin, 100 * solo.delta))
            print("              removing BOTH gives +%.1f%%. The coverage was real," % (
                100 * joint.delta))
            print("              just already paid for. Drop '%s', keep '%s'." % (
                name, twin))
        if not twins:
            print("              No masking partner found: it contributes nothing at all.")
    if not dead:
        print("  Every layer contributed measurably. (Unexpected -- check the seed.)")

    head("5. Harness controls -- does this thing have any power?")
    for label, sut in (("undefended control (should be ~100%)", UndefendedControl()),
                       ("hard-refusal baseline (should be 0%)", HardRefusalBaseline())):
        res = run(sut, corpus, frozenset(DEFENCES))
        print("  %-38s ASR %s" % (label, res.asr().fmt()))
    print()
    print("  Without these, 'low ASR' is ambiguous between a safe system and a")
    print("  harness too weak to land an attack on anything.")

    head("Verdict")
    print("  - Headline '%.0f%% blocked' is misleading: %s sits at %.0f%%." % (
        100 * rep.headline_block_rate, rep.worst_category,
        100 * rep.per_category[rep.worst_category].point))
    print("  - %s can be removed with no measured ASR cost; it saves %.0f ms." % (
        ", ".join(dead) if dead else "(none)",
        sum(DEFENCE_LATENCY_MS[d] for d in dead)))
    best = max(singles, key=lambda r: r.delta)
    print("  - %s is the load-bearing layer: +%.1f%% ASR without it, at %.0f ms." % (
        next(iter(best.removed)), 100 * best.delta, best.latency_saved_ms))
    print("  - Nothing smaller than about %.1f%% was resolvable at %d trials/category." % (
        100 * max(rep.mdd_per_category.values()), TRIALS))
    print()


if __name__ == "__main__":
    main()

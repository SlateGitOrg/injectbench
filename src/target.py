"""The system under test, behind an interface, plus the defence pipeline.

THERE IS NO MODEL HERE. No weights, no Ollama, no GPU, no network API. The
backend is a deterministic simulator. It is not a stand-in for a real model's
behaviour and no number in this repo is a claim about any real model.

What it IS is a system with KNOWN ground truth, which is the only way to test
the part that actually matters: whether the harness's statistics and ablation
recover the truth. A harness validated against a real model cannot be
validated at all, because nobody knows the right answer.

`Target` is the seam. Swap in an implementation that calls a real system and
every measurement above this line is unchanged.
"""

import random
from typing import Iterable, Protocol, Set

from .corpus import Attack
from .taxonomy import (
    CUE_ENCODED_PAYLOAD,
    CUE_EXFIL_SHAPED_OUTPUT,
    CUE_OVERRIDE_PHRASE,
    CUE_PROVENANCE_MARGIN,
    CUE_UNTRUSTED_CHANNEL,
    DEFENCES,
    DEFENCE_LATENCY_MS,
    DELIMITER_ISOLATION,
    INPUT_FILTER,
    NORMALIZER,
    OUTPUT_SCANNER,
    PROFILES,
    PROVENANCE_TAGGER,
)


class Target(Protocol):
    """Anything that can be attacked. Returns True if the attack succeeded."""

    def attempt(self, attack: Attack, enabled_defences: Set[str]) -> bool:
        ...


def _blocks(defence: str, attack: Attack) -> bool:
    if defence == INPUT_FILTER:
        return attack.has(CUE_OVERRIDE_PHRASE)
    if defence == DELIMITER_ISOLATION:
        return attack.has(CUE_UNTRUSTED_CHANNEL)
    if defence == PROVENANCE_TAGGER:
        # Strict subset of DELIMITER_ISOLATION by construction. See the note in
        # taxonomy.CategoryProfile.residual_asr.
        return attack.has(CUE_UNTRUSTED_CHANNEL) and attack.has(CUE_PROVENANCE_MARGIN)
    if defence == NORMALIZER:
        return attack.has(CUE_ENCODED_PAYLOAD)
    if defence == OUTPUT_SCANNER:
        return attack.has(CUE_EXFIL_SHAPED_OUTPUT)
    raise KeyError(defence)


class SimulatedTarget:
    """Deterministic simulated assistant.

    Susceptibility depends on the attack's taxonomy category (via the cue
    probabilities baked into the corpus) and on which defences are enabled.
    No randomness lives here at all: every draw happened once, in the corpus.
    Calling attempt() twice with the same arguments is bit-identical, and -
    critically - the SAME attack object is used across every ablation arm.
    """

    def attempt(self, attack: Attack, enabled_defences: Iterable[str]) -> bool:
        if not _would_comply(attack):
            return False
        for defence in enabled_defences:
            if defence not in DEFENCES:
                raise KeyError("unknown defence: %s" % defence)
            if _blocks(defence, attack):
                return False
        return True


def _would_comply(attack: Attack) -> bool:
    """Would the target comply with nothing in front of it?

    Derived from the attack id deterministically rather than stored as a cue,
    so that base susceptibility and defence-visible traces are drawn from
    independent streams. If they shared a stream, a defence would appear to
    work simply because attacks it catches are attacks that were never going
    to land.
    """
    rng = random.Random("base|%s" % attack.attack_id)
    return rng.random() < PROFILES[attack.category].base_success


class UndefendedControl:
    """Positive control: complies with everything, defences are ignored.

    Required by the testing strategy. A red-team harness reporting low ASR
    because it is weak looks identical to one reporting low ASR because the
    system is safe. Running against this proves the harness has power.
    """

    def attempt(self, attack: Attack, enabled_defences: Iterable[str]) -> bool:
        return True


class HardRefusalBaseline:
    """Negative control: refuses everything. Must measure ASR of exactly 0."""

    def attempt(self, attack: Attack, enabled_defences: Iterable[str]) -> bool:
        return False


def stack_latency_ms(enabled_defences: Iterable[str]) -> float:
    return sum(DEFENCE_LATENCY_MS[d] for d in enabled_defences)

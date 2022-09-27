from synth.syntax.grammars.enumeration.heap_search import (
    Bucket,
    enumerate_prob_grammar,
    enumerate_bucket_prob_grammar,
)
from synth.syntax.grammars.cfg import CFG
from synth.syntax.grammars.ttcfg import TTCFG
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar
from synth.syntax.dsl import DSL
from synth.syntax.type_system import (
    INT,
    STRING,
    FunctionType,
    List,
    PolymorphicType,
    PrimitiveType,
)

import pytest


syntax = {
    "+": FunctionType(INT, INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    "1": INT,
    "non_productive": FunctionType(INT, STRING),
}
dsl = DSL(syntax)
dsl.instantiate_polymorphic_types()
testdata = [
    CFG.depth_constraint(dsl, FunctionType(INT, INT), 3),
    TTCFG.size_constraint(dsl, FunctionType(INT, INT), 5),
]


@pytest.mark.parametrize("cfg", testdata)
def test_unicity_heapSearch(cfg: TTCFG) -> None:
    pcfg = ProbDetGrammar.uniform(cfg)
    seen = set()
    for program in enumerate_prob_grammar(pcfg):
        assert program not in seen
        seen.add(program)
    assert len(seen) == cfg.programs()


@pytest.mark.parametrize("cfg", testdata)
def test_order_heapSearch(cfg: TTCFG) -> None:
    pcfg = ProbDetGrammar.uniform(cfg)
    last = 1.0
    for program in enumerate_prob_grammar(pcfg):
        p = pcfg.probability(program)
        assert p <= last
        last = p


@pytest.mark.parametrize("cfg", testdata)
def test_unicity_bucketSearch(cfg: TTCFG) -> None:
    pcfg = ProbDetGrammar.uniform(cfg)
    for bucketSize in range(3, 10):
        seen = set()
        for program in enumerate_bucket_prob_grammar(pcfg, bucket_size=bucketSize):
            assert program not in seen
            seen.add(program)
        assert len(seen) == cfg.programs()


@pytest.mark.parametrize("cfg", testdata)
def test_order_bucketSearch(cfg: TTCFG) -> None:
    pcfg = ProbDetGrammar.uniform(cfg)
    for bucketSize in range(3, 10):
        last = Bucket(bucketSize)
        for program in enumerate_bucket_prob_grammar(pcfg, bucket_size=bucketSize):
            p = pcfg.reduce_derivations(
                lambda b, S, P, _: b.add_prob_uniform(pcfg.probabilities[S][P]),
                Bucket(bucketSize),
                program,
            )
            assert p.size == bucketSize
            assert p >= last or last == Bucket(bucketSize)
            last = p
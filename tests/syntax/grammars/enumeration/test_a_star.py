from synth.syntax.grammars.enumeration.a_star import (
    enumerate_prob_grammar,
)
from synth.syntax.grammars.cfg import CFG
from synth.syntax.grammars.ttcfg import TTCFG
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar
from synth.syntax.dsl import DSL
from synth.syntax.type_system import (
    INT,
    STRING,
    List,
    PolymorphicType,
    PrimitiveType,
)
from synth.syntax.type_helper import FunctionType, auto_type

import pytest


syntax = {
    "+": FunctionType(INT, INT, INT),
    "head": FunctionType(List(PolymorphicType("a")), PolymorphicType("a")),
    "non_reachable": PrimitiveType("non_reachable"),
    "1": INT,
    "2": INT,
    "non_productive": FunctionType(INT, STRING),
}
dsl = DSL(syntax)
dsl.instantiate_polymorphic_types()
testdata = [
    CFG.depth_constraint(dsl, FunctionType(INT, INT), 3),
    CFG.depth_constraint(dsl, FunctionType(INT, INT), 4),
]


@pytest.mark.parametrize("cfg", testdata)
def test_unicity_a_star(cfg: TTCFG) -> None:
    pcfg = ProbDetGrammar.uniform(cfg)
    seen = set()
    print(cfg)
    for program in enumerate_prob_grammar(pcfg):
        assert program not in seen
        seen.add(program)
    # print(pcfg.grammar)
    assert len(seen) == cfg.programs()


@pytest.mark.parametrize("cfg", testdata)
def test_order_a_star(cfg: TTCFG) -> None:
    pcfg = ProbDetGrammar.uniform(cfg)
    last = 1.0
    for program in enumerate_prob_grammar(pcfg):
        p = pcfg.probability(program)
        assert p <= last
        last = p


def test_infinite() -> None:
    pcfg = ProbDetGrammar.random(
        CFG.infinite(dsl, testdata[0].type_request, n_gram=1), 1
    )
    count = 10000
    last = 1.0
    for program in enumerate_prob_grammar(pcfg):
        count -= 1
        p = pcfg.probability(program)
        assert -1e-12 <= last - p, f"failed at program n°{count}:{program}"
        last = p
        if count < 0:
            break
    assert count == -1

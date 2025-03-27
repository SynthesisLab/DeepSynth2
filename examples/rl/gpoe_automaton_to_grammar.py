from typing import Dict, Set, Tuple
from synth.syntax.automata.tree_automaton import DFTA
from synth.syntax.dsl import DSL
from synth.syntax.grammars.enumeration.sbsur import enumerate_uniform_dfta
from synth.syntax.grammars.grammar import DerivableProgram
from synth.syntax.program import Constant, Primitive, Variable
from synth.syntax.type_helper import auto_type
from synth.syntax.type_system import Type


def parse(
    dsl: DSL,
    content: str,
    target_type_request: Type,
    constant_types: Set[Type],
    actions: int,
) -> DFTA[str, DerivableProgram]:
    lines = content.splitlines()
    finals = set(lines.pop(0)[len("finals:") :].split(","))
    terminals = lines.pop(0)[len("terminals:") :].split(",")
    nonterminals = lines.pop(0)[len("nonterminals:") :].split(",")
    rules: Dict[Tuple[DerivableProgram, Tuple[str, ...]], str] = {}

    action_states = set()

    state2type: dict[str, Type] = {}

    ACTION = auto_type("action")

    steps = len(lines) * len(lines)
    used = set()

    while lines and steps >= 0:
        steps -= 1
        line = lines.pop()
        elements = line.split(",")
        dst = elements.pop(0)
        primitive_name = elements.pop(0)
        if primitive_name.startswith("var"):
            if dst in state2type:
                var_type = state2type[dst]
                for i, arg in enumerate(target_type_request.arguments()):
                    if arg == var_type:
                        used.add(i)
                        rules[(Variable(i, arg), tuple())] = dst
                if var_type in constant_types:
                    rules[(Constant(var_type), tuple())] = dst
            else:
                lines.insert(0, line)
        else:
            if primitive_name[0] == "A":
                primitive = Primitive(primitive_name, ACTION)
            else:
                possibles_primitives = [
                    p for p in dsl.list_primitives if p.primitive == primitive_name
                ]
                if len(possibles_primitives) == 0:
                    assert False, f"could not parse primitive:'{primitive_name}'"
                elif len(possibles_primitives) == 1:
                    primitive = possibles_primitives.pop()
                else:
                    for i, arg in enumerate(elements):
                        arg_type = state2type.get(arg, None)
                        if arg_type is None:
                            continue
                        possibles_primitives = [
                            p
                            for p in possibles_primitives
                            if p.type.arguments()[i] == arg_type
                        ]
                        if len(possibles_primitives) <= 1:
                            break

                    if len(possibles_primitives) == 1:
                        primitive = possibles_primitives.pop()
                    else:
                        lines.insert(0, line)
            state2type[dst] = primitive.type.returns()
            args = elements
            for arg, arg_type in zip(args, primitive.type.arguments()):
                state2type[arg] = arg_type
            rules[(primitive, tuple(args))] = dst
            if primitive.type.returns().is_instance(ACTION):
                action_states.add(dst)
    assert len(lines) == 0, f"Unfinished parsing: {lines}"
    assert used == set(range(len(target_type_request.arguments()))), (
        "Some variables were not in the automaton and could not be added!"
    )
    dfta = DFTA(
        rules, action_states if actions > 0 else finals.difference(action_states)
    )
    dfta.reduce()
    # ACTIONS SPECIFIC TO CONTROL DSL
    if actions > 0:
        assert len(action_states) >= 1, f"action states: {action_states}"
        # Any of these states should work
        action_state = list(action_states).pop()
        for i in range(actions):
            dfta.rules[(Primitive(f"A{i}", ACTION), tuple())] = action_state
        # Action state must be final
        dfta.finals = {action_state}

        mapping = {k: action_state for k in action_states}

        dfta = dfta.map_states(lambda k: mapping.get(k, k))

    else:
        for key, dst in dfta.rules.copy().items():
            if dst in action_states:
                del dfta.rules[key]
    dfta.reduce()
    return dfta


if __name__ == "__main__":
    import sys

    with open(sys.argv[1]) as fd:
        content = fd.read()
    from control_dsl import get_dsl

    type_req = auto_type("float -> float -> action")
    dsl, eval = get_dsl(auto_type("float -> float -> float"), None)
    aut = parse(dsl, content, type_req, {}, 10)
    print(aut)
    enumerator = enumerate_uniform_dfta(aut)
    for program in enumerator.generator():
        print(program)
        eval.eval(program, [1, 2])

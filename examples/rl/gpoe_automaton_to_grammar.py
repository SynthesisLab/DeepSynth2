from typing import Dict, Generator, Set, Tuple
from synth.syntax.automata.tree_automaton import DFTA
from synth.syntax.dsl import DSL
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
    base_actions = set()

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
                continue
        else:
            if primitive_name[0] == "A":
                primitive = Primitive(primitive_name, ACTION)
                base_actions.add(dst)
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
                        continue
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

        mapping = {k: action_state for k in base_actions}

        dfta = dfta.map_states(lambda k: mapping.get(k, k))

    else:
        for key, dst in dfta.rules.copy().items():
            if dst in action_states:
                del dfta.rules[key]
    dfta.reduce()
    return dfta


def integer_partitions(k: int, n: int) -> Generator[Tuple[int, ...], None, None]:
    if k > n:
        return
    tup = [n - k + 1] + [1] * (k - 1)
    yield tuple(tup)
    while True:
        if tup[-1] == n - k + 1:
            break
        carry = tup[-1] - 1
        tup[-1] = 1
        for i in range(k - 2, -1, -1):
            if tup[i] > 1:
                break
        tup[i] -= 1
        tup[i + 1] += 1 + carry
        yield tuple(tup)


def size_constraint(
    dsl: DSL, type_request: Type, max_size: int
) -> DFTA[Tuple[int, Type], DerivableProgram]:
    rules = {}
    rtype = type_request.returns()

    finals = set()

    for i, var_type in enumerate(type_request.arguments()):
        is_final = var_type.is_instance(rtype)
        dst = (1, var_type)
        if is_final:
            finals.add(dst)
        rules[(Variable(i, var_type), tuple())] = dst

    for size in range(1, max_size + 1):
        for P in dsl.list_primitives:
            P_rtype = P.type.returns()
            args_types = P.type.arguments()
            is_final = P_rtype.is_instance(rtype)
            dst = (size, P_rtype)
            if is_final:
                finals.add(dst)
            if len(args_types) == 0:
                if size == 1:
                    rules[(P, tuple())] = dst
            else:
                for size_req in integer_partitions(len(args_types), size - 1):
                    args = tuple(zip(size_req, args_types))
                    rules[(P, args)] = dst
    dfta = DFTA(rules, finals)
    dfta.reduce()
    return dfta

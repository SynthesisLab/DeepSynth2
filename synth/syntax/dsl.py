import copy
import itertools
from typing import Any, Callable, Dict, Mapping, Optional, List as TList, Set, Tuple
from synth.syntax.type_helper import FunctionType

from synth.syntax.type_system import UNIT, Sum, Type, Arrow, List, UnknownType
from synth.syntax.program import Constant, Function, Primitive, Program, Variable


class DSL:
    """
    Object that represents a domain specific language

    Parameters:
    -----------
    - syntax: maps primitive names to their types
    - forbidden_patterns: forbidden local derivations

    """

    def __init__(
        self,
        syntax: Mapping[str, Type],
        forbidden_patterns: Optional[Dict[Tuple[str, int], Set[str]]] = None,
    ):
        self.list_primitives = [
            Primitive(primitive=p, type=t) for p, t in syntax.items()
        ]
        self.forbidden_patterns = forbidden_patterns or {}

    def __str__(self) -> str:
        s = "Print a DSL\n"
        for P in self.list_primitives:
            s = s + "{}: {}\n".format(P, P.type)
        return s

    def instantiate_polymorphic_types(self, upper_bound_type_size: int = 10) -> None:
        """
        Must be called before compilation into a grammar or parsing.
        Instantiate all polymorphic types.

        Parameters:
        -----------
        - upper_bound_type_size: maximum type size of type instantiated for polymorphic types
        """
        # Generate all basic types
        set_basic_types: Set[Type] = set()
        for P in self.list_primitives:
            set_basic_types_P, set_polymorphic_types_P = P.type.decompose_type()
            set_basic_types = set_basic_types | set_basic_types_P
        if UNIT in set_basic_types:
            set_basic_types.remove(UNIT)

        set_types = set(set_basic_types)
        for type_ in set_basic_types:
            # Instanciate List(x) and List(List(x))
            tmp_new_type = List(type_)
            set_types.add(tmp_new_type)
            set_types.add(List(tmp_new_type))
            # Instanciate Arrow(x, y)
            for type_2 in set_basic_types:
                new_type2 = Arrow(type_, type_2)
                set_types.add(new_type2)

        # Replace Primitive with Polymorphic types with their instanciated counterpart
        for P in self.list_primitives[:]:
            type_P = P.type
            _, set_polymorphic_types_P = type_P.decompose_type()
            if set_polymorphic_types_P:
                set_instantiated_types: Set[Type] = set()
                set_instantiated_types.add(type_P)
                for poly_type in set_polymorphic_types_P:
                    new_set_instantiated_types: Set[Type] = set()
                    for type_ in set_types:
                        if (
                            not poly_type.can_be(type_)
                            or type_.size() > upper_bound_type_size
                        ):
                            continue
                        for instantiated_type in set_instantiated_types:
                            unifier = {str(poly_type): type_}
                            intermediate_type = copy.deepcopy(instantiated_type)
                            new_type = intermediate_type.unify(unifier)
                            new_set_instantiated_types.add(new_type)
                    set_instantiated_types = new_set_instantiated_types
                for type_ in set_instantiated_types:
                    instantiated_P = Primitive(P.primitive, type=type_)
                    if instantiated_P not in self.list_primitives:
                        self.list_primitives.append(instantiated_P)
                self.list_primitives.remove(P)

        # Duplicate things for Sum types
        for P in self.list_primitives[:]:
            versions = P.type.all_versions()
            if len(versions) > 1:
                for type_ in versions:
                    instantiated_P = Primitive(P.primitive, type=type_)
                    self.list_primitives.append(instantiated_P)
                self.list_primitives.remove(P)

        # Now remove all UNIT as parameters from signatures
        for P in self.list_primitives[:]:
            if any(arg == UNIT for arg in P.type.arguments()):
                P.type = P.type.without_unit_arguments()

    def __eq__(self, o: object) -> bool:
        return isinstance(o, DSL) and set(self.list_primitives) == set(
            o.list_primitives
        )

    def fix_types(
        self,
        program: Program,
    ) -> Program:
        """
        Takes a program with possibly UnknownTypes anywhere and try to instantiate the types correctly.
        This does not solves type equations, it is much weaker but should be enough for most use cases.

        Parameters:
        -----------
        - program: the progam whose types needs fixing

        Returns:
        -----------
        A parsed program that matches the given string
        """
        return self.__fix_types__(program)[0]

    def __fix_types__(
        self,
        program: Program,
        forced_type: Optional[Type] = None,
        force_fix: bool = False,
    ) -> Tuple[Program, bool]:
        is_ambiguous = False
        if isinstance(program, Function):
            fixed_fun, ambiguous = self.__fix_types__(
                program.function, force_fix=force_fix
            )
            args = [
                self.__fix_types__(arg, arg_type)[0]
                for arg, arg_type in zip(program.arguments, fixed_fun.type.arguments())
            ]

            if ambiguous and forced_type is not None:
                print(
                    "before:",
                    fixed_fun,
                    "type:",
                    fixed_fun.type,
                    "args:",
                    args,
                    "target:",
                    FunctionType(*([arg.type for arg in args] + [forced_type])),
                )
                fixed_fun = self.__fix_types__(
                    program.function,
                    FunctionType(*[arg.type for arg in args], forced_type),
                    force_fix=force_fix,
                )[0]
                print("after:", fixed_fun, "type:", fixed_fun.type)
                args = [
                    self.__fix_types__(arg, arg_type, force_fix=force_fix)[0]
                    for arg, arg_type in zip(
                        program.arguments, fixed_fun.type.arguments()
                    )
                ]
            out: Program = Function(fixed_fun, args)
        elif not force_fix and not program.type.is_under_specified():
            out = program
        elif isinstance(program, Variable):
            out = Variable(program.variable, forced_type or program.type)
        elif isinstance(program, Constant):
            out = Constant(
                forced_type or program.type, program.value, program.has_value()
            )
        elif isinstance(program, Primitive):
            if forced_type is None:
                matching = [
                    p for p in self.list_primitives if p.primitive == program.primitive
                ]
                if len(matching) == 1:
                    forced_type = matching[0].type
                elif len(matching) > 1:
                    is_ambiguous = True
                    forced_type = Sum(*list(map(lambda x: x.type, matching)))
            out = Primitive(program.primitive, forced_type or program.type)
        else:
            assert False, "no implemented"
        return out, is_ambiguous

    def auto_parse_program(
        self,
        program: str,
        constants: Dict[str, Tuple[Type, Any]] = {},
    ) -> Program:
        """
        Parse a program from its string representation given the type request.
        It will try to automatically fix types to guess the type request.

        Parameters:
        -----------
        - program: the string representation of the program, i.e. str(prog)
        - constants: str representation of constants that map to their (type, value)

        Returns:
        -----------
        A parsed program that matches the given string
        """
        nvars = 0
        for s in program.split("var"):
            i = 0
            while i < len(s) and s[i].isdigit():
                i += 1
            if i > 0:
                nvars = max(int(s[:i]) + 1, nvars)
        tr = FunctionType(*[UnknownType()] * (nvars + 1))
        return self.fix_types(self.parse_program(program, tr, constants, False))

    def __parse_program__(
        self,
        program: str,
        type_request: Type,
        constants: Dict[str, Tuple[Type, Any]] = {},
    ) -> TList[Program]:
        """
        Produce all possible interpretations of a parsed program.
        """
        if " " in program:
            parts = list(
                map(
                    lambda p: self.__parse_program__(p, type_request, constants),
                    program.split(" "),
                )
            )
            function_calls: TList[int] = []
            level = 0
            levels: TList[int] = []
            elements = program.split(" ")
            for element in elements:
                if level > 0:
                    function_calls[levels[-1]] += 1
                function_calls.append(0)
                if element.startswith("("):
                    level += 1
                    levels.append(len(function_calls) - 1)
                end = 1
                while element[-end] == ")":
                    level -= 1
                    end += 1
                    levels.pop()

            n = len(parts)

            def parse_stack(i: int) -> TList[Tuple[Program, int]]:
                if i + 1 == n:
                    return [(p, n) for p in parts[-1]]
                current = parts[i]
                f_call = function_calls[i]
                out: TList[Tuple[Program, int]] = []
                for some in current:
                    if some.type.is_instance(Arrow) and f_call > 0:
                        poss_args: TList[Tuple[TList[Program], int]] = [([], i + 1)]
                        for _ in some.type.arguments()[:f_call]:
                            next = []
                            for poss, j in poss_args:
                                parsed = parse_stack(j)
                                for x, k in parsed:
                                    next.append((poss + [x], k))
                            poss_args = next

                        for poss, j in poss_args:
                            out.append((Function(some, list(poss)), j))
                    else:
                        out.append((some, i + 1))
                return out

            sols = parse_stack(0)

            return [p for p, _ in sols]
        else:
            program = program.strip("()")
            matching: TList[Program] = [
                P for P in self.list_primitives if P.primitive == program
            ]
            if len(matching) > 0:
                return matching
            elif program.startswith("var"):
                varno = int(program[3:])
                vart = type_request
                if type_request.is_instance(Arrow):
                    vart = type_request.arguments()[varno]
                return [Variable(varno, vart)]
            elif program in constants:
                t, val = constants[program]
                return [Constant(t, val, True)]
            assert False, f"can't parse: '{program}'"

    def parse_program(
        self,
        program: str,
        type_request: Type,
        constants: Dict[str, Tuple[Type, Any]] = {},
        check: bool = True,
    ) -> Program:
        """
        Parse a program from its string representation given the type request.

        Parameters:
        -----------
        - program: the string representation of the program, i.e. str(prog)
        - type_request: the type of the requested program in order to identify variable types
        - constants: str representation of constants that map to their (type, value)
        - check: ensure the program was correctly parsed with type checking

        Returns:
        -----------
        A parsed program that matches the given string
        """
        possibles = self.__parse_program__(program, type_request, constants)
        if check:
            coherents = [p for p in possibles if p.type_checks()]
            assert (
                len(coherents) > 0
            ), f"failed to parse a program that type checks for: {program}"
            return coherents[0]
        return possibles[0]

    def get_primitive(self, name: str) -> Optional[Primitive]:
        """
        Returns the Primitive object with the specified name if it exists and None otherwise

        Parameters:
        -----------
        - name: the name of the primitive to get
        """
        for P in self.list_primitives:
            if P.primitive == name:
                return P
        return None

    def instantiate_semantics(
        self, semantics: Dict[str, Callable]
    ) -> Dict[Primitive, Callable]:
        """
        Transform the semantics dictionnary from strings to primitives.
        """
        dico = {}
        for key, f in semantics.items():
            for p in self.list_primitives:
                if p.primitive == key:
                    dico[p] = f
        return dico

    def __or__(self, other: "DSL") -> "DSL":
        out = DSL({})
        out.list_primitives += self.list_primitives
        for prim in other.list_primitives:
            if prim not in self.list_primitives:
                out.list_primitives.append(prim)
        out.forbidden_patterns = {k: v for k, v in self.forbidden_patterns.items()}
        for k, v in other.forbidden_patterns.items():
            if k not in out.forbidden_patterns:
                out.forbidden_patterns[k] = v

        return out

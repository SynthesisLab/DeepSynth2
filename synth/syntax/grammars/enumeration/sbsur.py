from typing import (
    Generator,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import numpy as np


from synth.filter.filter import Filter
from synth.syntax.automata.tree_automaton import DFTA
from synth.syntax.grammars.enumeration.program_enumerator import ProgramEnumerator
from synth.syntax.grammars.tagged_u_grammar import ProbUGrammar
from synth.syntax.program import Function, Program
from synth.syntax.grammars.tagged_det_grammar import ProbDetGrammar

from sbsur import SequenceGenerator, sample

from synth.syntax.type_system import Type

U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


class SBSUR(
    ProgramEnumerator[None],
    Generic[U, V, W],
):
    def __init__(
        self,
        G: ProbDetGrammar[U, V, W],
        filter: Optional[Filter[Program]] = None,
        batch_size: int = 1,
    ) -> None:
        # G must be in log prob
        super().__init__(filter)
        self.current: Optional[Program] = None
        self.batch_size = batch_size

        self.G = G
        self.start = G.start
        self.rules = G.rules

        self.probs = {}
        self.mapping = {}

        for S, dico in sorted(G.rules.items()):
            if S not in self.probs:
                self.probs[S] = []
                self.mapping[S] = []
            for P in sorted(dico):
                self.probs[S].append(self.G.probabilities[S][P])
                self.mapping[S].append(P)

    def probability(self, program: Program) -> float:
        return self.G.probability(program)

    @classmethod
    def name(cls) -> str:
        return "sbsur"

    def __seq_to_prob__(self, seq: List[int]) -> Optional[List[float]]:
        current = self.G.start
        info = self.G.start_information()
        while seq:
            selected_index = seq.pop(0)
            P = self.mapping[current][selected_index]
            info, current = self.G.derive(info, current, P)

        out = self.probs.get(current, None)
        return out

    def __seq_to_prog__(self, seq: List[int]) -> Program:
        current = self.G.start
        info = self.G.start_information()
        elements = []
        while seq:
            selected_index = seq.pop(0)
            P = self.mapping[current][selected_index]
            elements.append((P, current[0]))
            info, current = self.G.derive(info, current, P)
        return self.__build__(elements)

    def __build__(self, elements: List[Tuple[Program, Type]]) -> Program:
        stack = []
        while elements:
            P, t = elements.pop()
            nargs = len(P.type.arguments())
            if nargs == 0:
                stack.append(P)
            else:
                args = stack[-nargs:]
                stack = stack[:-nargs]
                stack.append(Function(P, args))
        assert len(stack) == 1
        prog = stack.pop()
        return prog

    def generator(self) -> Generator[Program, None, None]:
        """
        A generator which outputs the next most probable program
        """
        max_categories: int = max(len(self.probs[S]) for S in self.probs)
        # since at any decision there is at most 2 choices
        seed: int = 0
        # Create a sequence generator, it can be used until you exhaust it i.e. you sampled everything
        gen: SequenceGenerator = SequenceGenerator(
            lambda sequences: [self.__seq_to_prob__(seq) for seq in sequences],
            max_categories,
            seed,
        )
        while True:
            batch = sample(gen, self.batch_size)
            if len(batch) < self.batch_size:
                break
            for el in batch:
                prog = self.__seq_to_prog__(el)
                if self.filter is not None and self.filter(prog):
                    continue

                yield prog

    def merge_program(self, representative: Program, other: Program) -> None:
        """
        Merge other into representative.
        In other words, other will no longer be generated through heap search
        """
        pass

    def programs_in_banks(self) -> int:
        return 0

    def programs_in_queues(self) -> int:
        return 0

    def clone(self, G: Union[ProbDetGrammar, ProbUGrammar]) -> "SBSUR[U, V, W]":
        assert isinstance(G, ProbDetGrammar)
        enum = self.__class__(G)
        return enum


def enumerate_prob_grammar(
    G: ProbDetGrammar[U, V, W], batch_size: int = 1
) -> SBSUR[U, V, W]:
    Gp: ProbDetGrammar = ProbDetGrammar(
        G.grammar,
        {
            S: {P: np.log(p) for P, p in val.items() if p > 0}
            for S, val in G.probabilities.items()
        },
    )
    return SBSUR(Gp, batch_size=batch_size)


class SBSURDFTA(
    ProgramEnumerator[None],
    Generic[U, V],
):
    def __init__(
        self,
        dfta: DFTA[U, V],
        filter: Optional[Filter[Program]] = None,
        batch_size: int = 1,
    ) -> None:
        super().__init__(filter)
        self.dfta = dfta
        self.starts = sorted(dfta.finals)
        self.batch_size = batch_size

        probs = {}
        self.mapping = {}

        for (P, args), dst in dfta.rules.items():
            if dst not in probs:
                probs[dst] = []
                self.mapping[dst] = []
            probs[dst].append(1)
            self.mapping[dst].append((P, args))

        self.probs = {}
        for P, elems in sorted(probs.items()):
            self.probs[P] = len(elems) * [np.log(1 / len(elems))]
        # start
        self.probs[None] = len(self.starts) * [np.log(1 / len(self.starts))]

    def probability(self, program: Program) -> float:
        return -1

    @classmethod
    def name(cls) -> str:
        return "sbsur-dfta"

    def __seq_to_prob__(self, seq: List[int]) -> Optional[List[float]]:
        # print("\tdecisions:", len(seq), seq, "so far=", self.__seq_to_elements__(seq[:]))
        current = None
        stack = []
        selected_start = False
        while seq:
            selected_index = seq.pop(0)
            if selected_start:
                P, args = self.mapping[current][selected_index]
                stack += args
                if len(stack) == 0:
                    return None
                current = stack.pop()
            else:
                current = self.starts[selected_index]
                selected_start = True

        out = self.probs.get(current, None)
        return out

    def __seq_to_elements__(self, seq: List[int]) -> List[Program]:
        print("decoding:", seq)
        current = None
        stack = [None]
        elements = []
        selected_start = False
        while seq:
            print("\tcurrent:", current, "elements:", elements, "stack:", stack)
            current = stack.pop()
            selected_index = seq.pop(0)
            if selected_start:
                P, args = self.mapping[current][selected_index]
                stack += args
                elements.append(P)
                # if len(stack) == 0:
                # return elements
            else:
                stack.append(self.starts[selected_index])
                selected_start = True

        return elements

    def __seq_to_prog__(self, seq: List[int]) -> Program:
        return self.__build__(self.__seq_to_elements__(seq))

    def __build__(self, elements: List[Program]) -> Program:
        print("building from:", elements)
        stack = []
        while elements:
            P = elements.pop()
            nargs = len(P.type.arguments())
            if nargs == 0:
                stack.append(P)
            else:
                args = stack[-nargs:]
                stack = stack[:-nargs]
                stack.append(Function(P, args))
        assert len(stack) == 1
        prog = stack.pop()
        return prog

    def generator(self) -> Generator[Program, None, None]:
        """
        A generator which outputs the next most probable program
        """
        max_categories: int = max(len(self.probs[S]) for S in self.probs)
        # since at any decision there is at most 2 choices
        seed: int = 0
        # Create a sequence generator, it can be used until you exhaust it i.e. you sampled everything
        gen: SequenceGenerator = SequenceGenerator(
            lambda sequences: [self.__seq_to_prob__(seq) for seq in sequences],
            max_categories,
            seed,
        )
        while True:
            batch = sample(gen, self.batch_size)
            if len(batch) < self.batch_size:
                break
            for el in batch:
                prog = self.__seq_to_prog__(el)
                if self.filter is not None and self.filter(prog):
                    continue

                yield prog

    def merge_program(self, representative: Program, other: Program) -> None:
        """
        Merge other into representative.
        In other words, other will no longer be generated through heap search
        """
        pass

    def programs_in_banks(self) -> int:
        return 0

    def programs_in_queues(self) -> int:
        return 0

    def clone(self, G: Union[ProbDetGrammar, ProbUGrammar]) -> "SBSURDFTA[U, V]":
        raise NotImplementedError


def enumerate_uniform_dfta(G: DFTA[U, V], batch_size: int = 1) -> SBSURDFTA[U, V]:
    return SBSURDFTA(G, batch_size=batch_size)

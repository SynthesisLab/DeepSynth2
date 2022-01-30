from functools import reduce
from typing import Callable, Dict, Iterable, List, Set, Tuple, Optional, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from synth.syntax.concrete.concrete_cfg import ConcreteCFG, Context
from synth.syntax.concrete.concrete_pcfg import ConcretePCFG
from synth.syntax.dsl import DSL
from synth.syntax.program import Function, Primitive, Program, Variable
from synth.syntax.type_system import Arrow, Type


LogPRules = Dict[Context, Dict[Program, Tuple[List[Context], Tensor]]]


class ConcreteLogPCFG:
    """
    Special version of ConcretePCFG to compute with Tensors
    """

    def __init__(
        self, start: Context, rules: LogPRules, max_program_depth: int, type_req: Type
    ):
        self.start = start
        self.rules = rules
        self.max_program_depth = max_program_depth
        self.hash_table_programs: Dict[int, Program] = {}

        self.hash = hash(str(rules))
        self.type_request = type_req

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, ConcreteLogPCFG)
            and self.type_request == o.type_request
            and self.rules == o.rules
        )

    def __str__(self) -> str:
        s = "Print a LogPCFG\n"
        s += "start: {}\n".format(self.start)
        for S in reversed(self.rules):
            s += "#\n {}\n".format(S)
            for P in self.rules[S]:
                args_P, w = self.rules[S][P]
                s += "   {} - {}: {}     {}\n".format(P, P.type, args_P, w.item())
        return s

    def __repr__(self) -> str:
        return self.__str__()

    def log_probability(self, P: Program, S: Optional[Context] = None) -> Tensor:
        """
        Compute the log probability of a program P generated from the non-terminal S
        """
        S = S or self.start
        if isinstance(P, Function):
            F = P.function
            args_P = P.arguments
            probability = self.rules[S][F][1]

            for i, arg in enumerate(args_P):
                probability += self.log_probability(arg, self.rules[S][F][0][i])
            return probability

        elif isinstance(P, (Variable, Primitive)):
            return self.rules[S][P][1]

        print("log_probability_program", P)
        assert False

    def to_pcfg(self) -> ConcretePCFG:
        rules = {
            S: {P: (args, np.exp(w.item())) for P, (args, w) in self.rules[S].items()}
            for S in self.rules
        }
        return ConcretePCFG(self.start, rules, self.max_program_depth, True)


class BigramsPredictorLayer(nn.Module):
    """

    Parameters:
    ------------
    - input_size: int - the input size of the tensor to this layer
    - cfgs: Iterable[ConcreteCFG] - the set of all supported CFG
    - variable_probability: float = 0.2 - the probability mass of all variable at any given derivation level
    """

    def __init__(
        self,
        input_size: int,
        cfgs: Iterable[ConcreteCFG],
        variable_probability: float = 0.2,
    ):
        super(BigramsPredictorLayer, self).__init__()

        self.cfg_dictionary = {cfg.type_request: cfg for cfg in cfgs}
        self.variable_probability = variable_probability

        # List all primitives and variables used in our CFGs
        primitives_list: Set[Union[Primitive, Variable]] = reduce(
            lambda acc, el: acc | el,
            [set(v.keys()) for cfg in cfgs for v in cfg.rules.values()],
            set(),
        )
        # map primitives_list => int
        self.symbol2index = {
            symbol: index
            for index, symbol in enumerate(
                prim for prim in primitives_list if not isinstance(prim, Variable)
            )
        }
        # list all primitives and variables that are functions
        func_primitives: List[Union[Primitive, Variable]] = [
            p for p in primitives_list if isinstance(p.type, Arrow)
        ]
        self.parent2index = {
            symbol: index for index, symbol in enumerate(func_primitives)
        }

        # IMPORTANT: we do not predict variables!
        self.number_of_primitives = len(self.symbol2index)
        self.number_of_parents = len(self.parent2index) + 1  # could be None
        self.maximum_arguments = max(
            len(p.type.arguments()) if isinstance(p.type, Arrow) else 0
            for p in func_primitives
        )
        self.log_probs_predictor = nn.Linear(
            input_size,
            self.number_of_parents * self.maximum_arguments * self.number_of_primitives,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        batch_IOs is a tensor of size
        (batch_size, input_size)

        returns: (batch_size, self.number_of_parents, self.maximum_arguments, self.number_of_primitives)
        """
        y: Tensor = self.log_probs_predictor(x)
        z = F.log_softmax(
            y.view(
                (
                    -1,
                    self.number_of_parents,
                    self.maximum_arguments,
                    self.number_of_primitives,
                )
            ),
            dim=-1,
        )
        return z

    def tensor2pcfg(
        self,
        x: Tensor,
        type_request: Type,
        total_variable_order: bool = True,
    ) -> ConcreteLogPCFG:
        """

        Parameters:
        ------------
        - x: Tensor - the tensor to be transformed into a PCFG
        - type_request: Type - the type request of the PCFG
        - total_variable_order: bool = True - reduce very slighlty (1e-7) some variable probabilities to ensure they are totally ordered in terms of probablities

        """
        device = x.device
        cfg = self.cfg_dictionary[type_request]
        rules: LogPRules = {}
        for S in cfg.rules:
            rules[S] = {}
            # Compute parent_index and argument_number
            if S.predecessors:
                parent_index = self.parent2index[S.predecessors[0][0]]
                argument_number = S.predecessors[0][1]
            else:
                parent_index = len(self.parent2index)  # no parent => None
                argument_number = 0
            # List of all variables derivable from S
            variables: List[Variable] = []
            # For each derivation parse probabilities
            for P in cfg.rules[S]:
                cpy_P = P
                if isinstance(P, Primitive):
                    primitive_index = self.symbol2index[P]
                    rules[S][cpy_P] = (
                        cfg.rules[S][P],
                        x[parent_index, argument_number, primitive_index],
                    )
                else:
                    V: Variable = P  # ensure typing
                    variables.append(V)
                    # All variables together have probability mass self.variable_probability
                    # then the probability of selecting a variable is uniform
            # If there are variables we need to normalise
            total = sum(np.exp(rules[S][P][1].item()) for P in rules[S])
            if variables:
                var_probability = self.variable_probability
                if total > 0:
                    # Normalise rest
                    to_add: float = np.log((1 - self.variable_probability) / total)
                    for O in rules[S]:
                        rules[S][O] = rules[S][O][0], rules[S][O][1] + to_add
                else:
                    # There are no other choices than variables
                    var_probability = 1
                # Normalise variable probability
                normalised_variable_logprob: float = np.log(
                    var_probability / len(variables)
                )
                for P in variables:
                    rules[S][P] = cfg.rules[S][P], torch.tensor(
                        normalised_variable_logprob
                    ).to(device)
                    # Trick to allow a total ordering on variables
                    if total_variable_order:
                        normalised_variable_logprob = np.log(
                            np.exp(normalised_variable_logprob) - 1e-7
                        )
            else:
                # We still need to normalise probabilities
                # Since all derivations aren't possible
                to_add = np.log(1 / total)
                for O in rules[S]:
                    rules[S][O] = rules[S][O][0], rules[S][O][1] + to_add
        grammar = ConcreteLogPCFG(cfg.start, rules, cfg.max_program_depth, type_request)
        return grammar


class ExactBigramsPredictorLayer(nn.Module):
    """
    Needs a lot less parameters than BigramsPredictorLayer but seems to have worse performance.

    Parameters:
    ------------
    - input_size: int - the input size of the tensor to this layer
    - cfgs: Iterable[ConcreteCFG] - the set of all supported CFG
    - variable_probability: float = 0.2 - the probability mass of all variable at any given derivation level
    """

    def __init__(
        self,
        input_size: int,
        cfgs: Iterable[ConcreteCFG],
        variable_probability: float = 0.2,
    ):
        super(ExactBigramsPredictorLayer, self).__init__()

        self.cfg_dictionary = {cfg.type_request: cfg for cfg in cfgs}
        self.variable_probability = variable_probability

        # Compute all pairs (S, P) where S has lost depth information
        self.all_pairs: Dict[
            Optional[Tuple[Union[Primitive, Variable], int]], Set[Primitive]
        ] = {}
        for cfg in cfgs:
            for S in cfg.rules:
                key = S.predecessors[0] if S.predecessors else None
                if not key in self.all_pairs:
                    self.all_pairs[key] = set()
                for P in cfg.rules[S]:
                    if not isinstance(P, Variable):
                        self.all_pairs[key].add(P)

        output_size = sum(len(self.all_pairs[S]) for S in self.all_pairs)

        self.s2index: Dict[
            Optional[Tuple[Union[Primitive, Variable], int]],
            Tuple[int, int, Dict[Primitive, int]],
        ] = {}
        current_index = 0
        for key, set_for_key in self.all_pairs.items():
            self.s2index[key] = (
                current_index,
                len(set_for_key),
                {P: i for i, P in enumerate(self.all_pairs[key])},
            )
            current_index += len(set_for_key)

        self.log_probs_predictor = nn.Linear(
            input_size,
            output_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        batch_IOs is a tensor of size
        (batch_size, input_size)

        returns: (batch_size, self.number_of_parents, self.maximum_arguments, self.number_of_primitives)
        """
        y: Tensor = self.log_probs_predictor(x)
        z = torch.ones_like(y)
        for _, (start, length, _) in self.s2index.items():
            z[:, start : start + length] = F.log_softmax(
                y[:, start : start + length], dim=-1
            )

        return z

    def tensor2pcfg(
        self,
        x: Tensor,
        type_request: Type,
        total_variable_order: bool = True,
    ) -> ConcreteLogPCFG:
        """

        Parameters:
        ------------
        - x: Tensor - the tensor to be transformed into a PCFG
        - type_request: Type - the type request of the PCFG
        - total_variable_order: bool = True - reduce very slighlty (1e-7) some variable probabilities to ensure they are totally ordered in terms of probablities

        """
        device = x.device
        cfg = self.cfg_dictionary[type_request]
        rules: LogPRules = {}
        for S in cfg.rules:
            rules[S] = {}
            key = S.predecessors[0] if S.predecessors else None
            start, length, symbol2index = self.s2index[key]
            y = x[start : start + length]

            # List of all variables derivable from S
            variables: List[Variable] = []
            # For each derivation parse probabilities
            for P in cfg.rules[S]:
                cpy_P = P
                if isinstance(P, Primitive):
                    primitive_index = symbol2index[P]
                    rules[S][cpy_P] = (
                        cfg.rules[S][P],
                        y[primitive_index],
                    )
                else:
                    V: Variable = P  # ensure typing
                    variables.append(V)
                    # All variables together have probability mass self.variable_probability
                    # then the probability of selecting a variable is uniform
            # If there are variables we need to normalise
            total = sum(np.exp(rules[S][P][1].item()) for P in rules[S])
            if variables:
                var_probability = self.variable_probability
                if total > 0:
                    # Normalise rest
                    to_add: float = np.log((1 - self.variable_probability) / total)
                    for O in rules[S]:
                        rules[S][O] = rules[S][O][0], rules[S][O][1] + to_add
                else:
                    # There are no other choices than variables
                    var_probability = 1
                # Normalise variable probability
                normalised_variable_logprob: float = np.log(
                    var_probability / len(variables)
                )
                for P in variables:
                    rules[S][P] = cfg.rules[S][P], torch.tensor(
                        normalised_variable_logprob
                    ).to(device)
                    # Trick to allow a total ordering on variables
                    if total_variable_order:
                        normalised_variable_logprob = np.log(
                            np.exp(normalised_variable_logprob) - 1e-7
                        )
            else:
                # We still need to normalise probabilities
                # Since all derivations aren't possible
                to_add = np.log(1 / total)
                for O in rules[S]:
                    rules[S][O] = rules[S][O][0], rules[S][O][1] + to_add
        grammar = ConcreteLogPCFG(cfg.start, rules, cfg.max_program_depth, type_request)
        return grammar


def loss_negative_log_prob(
    programs: Iterable[Program],
    log_pcfgs: Iterable[ConcreteLogPCFG],
    reduce: Optional[Callable[[Tensor], Tensor]] = torch.mean,
) -> Tensor:
    log_prob_list = [
        log_pcfg.log_probability(p) for p, log_pcfg in zip(programs, log_pcfgs)
    ]
    out = -torch.stack(log_prob_list)
    if reduce:
        out = reduce(out)
    return out

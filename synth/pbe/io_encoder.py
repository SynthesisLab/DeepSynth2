from typing import Any, List, Tuple

import torch
from torch import Tensor

from synth.nn.spec_encoder import SpecificationEncoder
from synth.specification import PBE
from synth.task import Task


class IOEncoder(SpecificationEncoder[PBE, Tensor]):
    def __init__(self, output_dimension: int, lexicon: List[Any]) -> None:
        self.output_dimension = output_dimension

        self.specialSymbols = [
            "STARTING",  # start of entire sequence
            "ENDOFINPUT",  # delimits the ending of an input - we might have multiple inputs
            "STARTOFOUTPUT",  # begins the start of the output
            "ENDING",  # ending of entire sequence
            "STARTOFLIST",
            "ENDOFLIST",
        ]
        self.lexicon = lexicon + self.specialSymbols
        self.lexicon_size = len(lexicon)
        self.symbol2index = {symbol: index for index, symbol in enumerate(self.lexicon)}
        self.starting_index = self.symbol2index["STARTING"]
        self.end_of_input_index = self.symbol2index["ENDOFINPUT"]
        self.start_of_output_index = self.symbol2index["STARTOFOUTPUT"]
        self.ending_index = self.symbol2index["ENDING"]
        self.start_list_index = self.symbol2index["STARTOFLIST"]
        self.end_list_index = self.symbol2index["ENDOFLIST"]

    def __encode_element__(self, x: Any, encoding: List[int]) -> None:
        if isinstance(x, List):
            encoding.append(self.start_list_index)
            for el in x:
                self.__encode_element__(el, encoding)
            encoding.append(self.end_list_index)
        else:
            encoding.append(self.symbol2index[x])

    def encode_IO(self, IO: Tuple[List, Any]) -> Tensor:
        """
        embed a list of inputs and its associated output
        IO is of the form [[I1, I2, ..., Ik], O]
        where I1, I2, ..., Ik are inputs and O is an output

        outputs a tensor of dimension self.output_dimension
        """
        e = [self.starting_index]
        inputs, output = IO
        for x in inputs:
            self.__encode_element__(x, e)
            e.append(self.end_of_input_index)
        e.append(self.start_of_output_index)
        self.__encode_element__(output, e)
        e.append(self.ending_index)
        size = len(e)
        if size > self.output_dimension:
            assert (
                False
            ), "IOEncodings.py: VariableSizeEncoding: IO too large: {} > {} for {}".format(
                size, self.output_dimension, IO
            )
        else:
            for _ in range(self.output_dimension - size - 1):
                e.append(self.ending_index)
        res = torch.LongTensor(e)
        return res

    def encode(self, task: Task[PBE]) -> Tensor:
        return torch.stack(
            [
                self.encode_IO((ex.inputs, ex.output))
                for ex in task.specification.examples
            ]
        )

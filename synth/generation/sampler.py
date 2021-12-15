from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List as TList, Optional, Tuple, TypeVar, Union
import copy

import numpy as np
import vose

from synth.syntax.type_system import List, Type

T = TypeVar("T")
U = TypeVar("U")


class Sampler(ABC, Generic[T]):
    @abstractmethod
    def sample(self, type: Type, **kwargs: Any) -> T:
        pass


class LexiconSampler(Sampler[U]):
    def __init__(
        self,
        lexicon: TList[U],
        probabilites: Optional[TList[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.lexicon = copy.deepcopy(lexicon)
        filled_probabilities = probabilites or [1 / len(self.lexicon) for _ in lexicon]
        self.sampler = vose.Sampler(np.array(filled_probabilities), seed)

    def sample(self, type: Type, **kwargs: Any) -> U:
        index: int = self.sampler.sample()
        return self.lexicon[index]


class ListSampler(Sampler[TList]):
    def __init__(
        self,
        element_sampler: Sampler[U],
        probabilities: Union[TList[float], TList[Tuple[int, float]]],
        max_depth: int = -1,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.element_sampler = element_sampler
        correct_prob: TList[Tuple[int, float]] = probabilities  # type: ignore
        if not isinstance(probabilities[0], tuple):
            correct_prob = [(i + 1, p) for i, p in enumerate(probabilities)]  # type: ignore
        self._length_mapping = [n for n, _ in correct_prob]
        self.sampler = vose.Sampler(np.array([p for _, p in correct_prob]), seed=seed)

    def sample(self, type: Type, **kwargs: Any) -> TList:
        assert self.max_depth < 0 and type.depth() <= self.max_depth
        assert isinstance(type, List)
        sampler: Sampler = self
        if not isinstance(type.element_type, List):
            sampler = self.element_sampler
        length: int = self._length_mapping[self.sampler.sample()]
        return [sampler.sample(type.element_type, **kwargs) for _ in range(length)]


class UnionSampler(Sampler[Any]):
    def __init__(self, samplers: Dict[Type, Sampler]) -> None:
        super().__init__()
        self.samplers = samplers

    def sample(self, type: Type, **kwargs: Any) -> Any:
        return self.samplers[type].sample(**kwargs)

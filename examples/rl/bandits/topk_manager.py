from examples.rl.program_evaluator import ProgramEvaluator
import numpy as np

from typing import List, Optional, Tuple, TypeVar, Generic


T = TypeVar("T", covariant=True)


class TopkManager(Generic[T]):
    #
    def __init__(
        self,
        evaluator: ProgramEvaluator,
        c: float = 0.7,
        k: int = 2,
    ) -> None:
        self.evaluator = evaluator
        self.candidates: List[T] = []
        self.k = k
        self.c = c

    def num_candidates(self) -> int:
        return len(self.candidates)

    def challenge_with(
        self,
        new_candidate: T,
        max_budget: int = 100,
        prior_experience: List[float] = [],
    ) -> Tuple[Optional[T], int]:
        """
        return: the T ejected and the no of calls to get_return
        """
        # Add new program
        self.evaluator.add_returns(new_candidate, prior_experience)
        self.candidates.append(new_candidate)
        ejected_candidate, budget_used = self.__run_until_ejection__(max_budget)
        if ejected_candidate:
            self.__eject__(ejected_candidate)
        return ejected_candidate, budget_used

    def get_best_stats(self) -> Tuple[T, float, float, float, float]:
        best_arm = np.argmax([self.evaluator.mean_return(p) for p in self.candidates])
        candidate = self.candidates[best_arm]
        n = self.evaluator.samples(candidate)
        if n == 0:
            return candidate, float("nan"), float("inf"), -float("inf"), float("inf")
        rew = self.evaluator.returns(candidate)
        mean_return = np.mean(rew)
        return (
            candidate,
            mean_return,
            n,
            min(rew),
            max(rew),
        )

    def run_at_least(self, min_budget: int, min_score: float = -float("inf")) -> int:
        best_arm = np.argmax([self.evaluator.mean_return(p) for p in self.candidates])
        candidate = self.candidates[best_arm]
        initial: int = self.evaluator.samples(candidate)
        budget_used: int = 0
        while (
            initial + budget_used < min_budget
            and self.evaluator.mean_return(candidate) >= min_score
        ):
            budget_used += 1
            has_no_error = self.evaluator.eval(candidate)
            if not has_no_error:
                break
        return budget_used

    def __run_until_ejection__(self, max_budget: int) -> Tuple[Optional[T], int]:
        """
        return: the T ejected and the cost
        """
        budget_used: int = 0
        while self.__get_candidate_to_eject__() is None and budget_used < max_budget:
            index: int = np.argmin([self.evaluator.samples(p) for p in self.candidates])
            candidate: T = self.candidates[index]
            has_no_error = self.evaluator.eval(candidate)
            if not has_no_error:
                return candidate, budget_used
            budget_used += 1
        return self.__get_candidate_to_eject__(
            len(self.candidates) >= self.k
        ), budget_used

    def __get_candidate_to_eject__(self, force: bool = False) -> Optional[T]:
        if len(self.candidates) == 1:
            return None
        mean_returns = [self.evaluator.mean_return(p) for p in self.candidates]
        worst_arm = np.argmin(mean_returns)
        worst = self.candidates[worst_arm]
        if force:
            return worst
        best_arm = np.argmax(mean_returns)
        best = self.candidates[best_arm]

        if mean_returns[best_arm] - self.uncertainty(best) >= mean_returns[
            worst_arm
        ] + self.uncertainty(worst):
            return worst
        return None

    def uncertainty(self, candidate: T) -> float:
        n = self.evaluator.samples(candidate)
        if n == 0:
            return float("inf")
        return self.c * np.sqrt(
            np.log(sum(self.evaluator.samples(p) for p in self.candidates)) / n
        )

    def __eject__(self, candidate: T):
        self.evaluator.delete_data(candidate)
        self.candidates.remove(candidate)

from ..cases import QuadraticSimulator, QuadraticProblem
from ..cases import RosenbrockSimulator, RosenbrockProblem

from octras.algorithms import RandomWalk
from octras import Loop, Evaluator

import pytest

def test_random_walk():
    evaluator = Evaluator(
        simulator = QuadraticSimulator(),
        problem = QuadraticProblem([2.0, 1.0])
    )

    for seed in (1000, 2000, 3000, 4000):
        algorithm = RandomWalk(evaluator, seed = seed)

        assert Loop(threshold = 1e-2).run(
            evaluator = evaluator,
            algorithm = algorithm
        ) == pytest.approx((2.0, 1.0), 1e-1)

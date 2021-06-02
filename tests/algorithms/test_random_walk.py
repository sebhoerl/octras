from ..cases.quadratic import QuadraticSimulator, QuadraticProblem
from ..cases.traffic import TrafficSimulator, TrafficProblem

from octras.algorithms import RandomWalk
from octras import Loop, Evaluator

import pytest

def test_random_walk_quadratic():
    problem = QuadraticProblem([2.0, 1.0])

    evaluator = Evaluator(
        simulator = QuadraticSimulator(),
        problem = problem
    )

    algorithm = RandomWalk(problem,
        seed = 1000
    )

    assert Loop(threshold = 1e-2).run(
        evaluator = evaluator,
        algorithm = algorithm
    ) == pytest.approx((2.0, 1.0), 1e-1)

def test_random_walk_traffic():
    problem = TrafficProblem()

    evaluator = Evaluator(
        simulator = TrafficSimulator(),
        problem = problem
    )

    algorithm = RandomWalk(problem,
        seed = 1000
    )

    assert Loop(threshold = 10.0).run(
        evaluator = evaluator,
        algorithm = algorithm
    ) == pytest.approx((510.0, 412.0), 10.0)

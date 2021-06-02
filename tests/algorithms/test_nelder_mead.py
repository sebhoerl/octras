from ..cases.quadratic import QuadraticSimulator, QuadraticProblem
from ..cases.traffic import TrafficSimulator, TrafficProblem

from octras.algorithms import NelderMead
from octras import Loop, Evaluator

import pytest

def test_nelder_mead_quadratic():
    problem = QuadraticProblem([2.0, 1.0], [0.0, 0.0])

    evaluator = Evaluator(
        simulator = QuadraticSimulator(),
        problem = problem
    )

    algorithm = NelderMead(problem)

    assert Loop(threshold = 1e-4).run(
        evaluator = evaluator,
        algorithm = algorithm
    ) == pytest.approx((2.0, 1.0), 1e-2)

def test_nelder_mead_traffic():
    problem = TrafficProblem()
    problem.use_bounds([[400.0, 600.0]] * 2)

    evaluator = Evaluator(
        simulator = TrafficSimulator(),
        problem = TrafficProblem()
    )

    algorithm = NelderMead(problem)

    assert Loop(threshold = 1e-2).run(
        evaluator = evaluator,
        algorithm = algorithm
    ) == pytest.approx((510.0, 412.0), 1.0)

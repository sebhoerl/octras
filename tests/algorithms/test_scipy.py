from ..cases.quadratic import QuadraticSimulator, QuadraticProblem
from ..cases.traffic import TrafficSimulator, TrafficProblem

from octras.algorithms import ScipyAlgorithm
from octras import Loop, Evaluator

import pytest

def test_cobyla_quadratic():
    problem = QuadraticProblem([2.0, 1.0], [0.0, 0.0])

    evaluator = Evaluator(
        simulator = QuadraticSimulator(),
        problem = problem
    )

    algorithm = ScipyAlgorithm(problem, method = "COBYLA")

    assert Loop(threshold = 1e-4).run(
        evaluator = evaluator,
        algorithm = algorithm
    ) == pytest.approx((2.0, 1.0), 1e-2)

def test_scipy_traffic():
    problem = TrafficProblem()

    evaluator = Evaluator(
        simulator = TrafficSimulator(),
        problem = problem
    )

    algorithm = ScipyAlgorithm(problem, method = "Nelder-Mead")

    assert Loop(threshold = 5.0).run(
        evaluator = evaluator,
        algorithm = algorithm
    ) == pytest.approx((510.0, 412.0), 1.0)

from ..cases.quadratic import QuadraticSimulator, QuadraticProblem
from ..cases.traffic import TrafficSimulator, TrafficProblem

from octras.algorithms import FDSA
from octras import Loop, Evaluator

import pytest

def test_fdsa_quadratic():
    problem = QuadraticProblem([2.0, 1.0], [0.0, 0.0])

    evaluator = Evaluator(
        simulator = QuadraticSimulator(),
        problem = problem
    )

    algorithm = FDSA(problem,
        perturbation_factor = 2e-2,
        gradient_factor = 0.2,
    )

    assert Loop(threshold = 1e-4).run(
        evaluator = evaluator,
        algorithm = algorithm
    ) == pytest.approx((2.0, 1.0), 1e-2)

def test_fdsa_traffic():
    problem = TrafficProblem()

    evaluator = Evaluator(
        simulator = TrafficSimulator(),
        problem = problem
    )

    algorithm = FDSA(problem,
        perturbation_factor = 10.0,
        gradient_factor = 1.0,
    )

    assert Loop(threshold = 1e-2).run(
        evaluator = evaluator,
        algorithm = algorithm
    ) == pytest.approx((510.0, 412.0), 1.0)

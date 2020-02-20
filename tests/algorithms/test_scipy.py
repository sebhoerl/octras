from ..cases import QuadraticSimulator, QuadraticProblem
from ..cases import RosenbrockSimulator, RosenbrockProblem

from octras.algorithms import ScipyAlgorithm
from octras import Loop, Evaluator

import pytest

def test_scipy():
    evaluator = Evaluator(
        simulator = QuadraticSimulator(),
        problem = QuadraticProblem([2.0, 1.0], [0.0, 0.0])
    )

    algorithm = ScipyAlgorithm(evaluator, method = "COBYLA")

    assert Loop(threshold = 1e-3).run(
        evaluator = evaluator,
        algorithm = algorithm
    ) == pytest.approx((2.0, 1.0), 1e-3)

from ..cases import QuadraticSimulator, QuadraticProblem
from ..cases import RosenbrockSimulator, RosenbrockProblem

from octras.algorithms import FDSA
from octras import Loop, Evaluator

import pytest

def test_fdsa():
    evaluator = Evaluator(
        simulator = QuadraticSimulator(),
        problem = QuadraticProblem([2.0, 1.0], [0.0, 0.0])
    )

    algorithm = FDSA(evaluator,
        perturbation_factor = 2e-2,
        gradient_factor = 0.2
    )

    assert Loop(threshold = 1e-4).run(
        evaluator = evaluator,
        algorithm = algorithm
    ) == pytest.approx((2.0, 1.0), 1e-2)

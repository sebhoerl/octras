from ..cases import QuadraticSimulator, QuadraticProblem
from ..cases import RosenbrockSimulator, RosenbrockProblem

from octras.algorithms import NelderMead
from octras import Loop, Evaluator

import pytest
import numpy as np

def test_nelder_mead():
    evaluator = Evaluator(
        simulator = QuadraticSimulator(),
        problem = QuadraticProblem([2.0, 1.0], [0.0, 0.0])
    )

    algorithm = NelderMead(evaluator)

    assert Loop(threshold = 1e-4).run(
        evaluator = evaluator,
        algorithm = algorithm
    ) == pytest.approx((2.0, 1.0), 1e-2)

def test_nelder_mead_congestion():
    evaluator = Evaluator(
        simulator = RosenbrockSimulator(),
        problem = RosenbrockProblem(2)
    )

    algorithm = NelderMead(evaluator, bounds = [
        [-6.0, 6.0],
        [-6.0, 6.0]
    ])

    assert Loop(threshold = 1e-6).run(
        evaluator = evaluator,
        algorithm = algorithm
    ) == pytest.approx((1.0, 1.0), 1e-2)

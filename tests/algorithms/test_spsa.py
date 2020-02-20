from ..cases import QuadraticSimulator, QuadraticProblem
from ..cases import RosenbrockSimulator, RosenbrockProblem

from octras.algorithms import SPSA
from octras import Loop, Evaluator

import pytest

def test_spsa():
    evaluator = Evaluator(
        simulator = QuadraticSimulator(),
        problem = QuadraticProblem([2.0, 1.0], [0.0, 0.0])
    )

    for seed in (1000, 2000, 3000, 4000):
        algorithm = SPSA(evaluator,
            perturbation_factor = 2e-2,
            gradient_factor = 0.2,
            seed = seed
        )

        assert Loop(threshold = 1e-4).run(
            evaluator = evaluator,
            algorithm = algorithm
        ) == pytest.approx((2.0, 1.0), 1e-2)

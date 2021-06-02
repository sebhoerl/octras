from ..cases.quadratic import QuadraticSimulator, QuadraticProblem
from ..cases.traffic import TrafficSimulator, TrafficProblem

from octras.algorithms import Opdyts
from octras import Loop, Evaluator

import pytest

def test_opdyts_traffic():
    problem = TrafficProblem(iterations = 40)

    evaluator = Evaluator(
        simulator = TrafficSimulator(),
        problem = problem
    )

    for seed in (1000, 2000, 3000, 4000):
        algorithm = Opdyts(problem,
            candidate_set_size = 16,
            number_of_transitions = 10,
            perturbation_length = 50,
            seed = seed
        )

        assert Loop(threshold = 1e-2).run(
            evaluator = evaluator,
            algorithm = algorithm
        ) == pytest.approx((510.0, 412.0), 1.0)

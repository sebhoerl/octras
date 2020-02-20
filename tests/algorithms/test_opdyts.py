from ..cases import CongestionSimulator, CongestionProblem

from octras.algorithms import Opdyts
from octras import Loop, Evaluator

import pytest
import numpy as np

def test_opdyts():
    loop = Loop()

    evaluator = Evaluator(
        simulator = CongestionSimulator(),
        problem = CongestionProblem(0.3, iterations = 10)
    )

    algorithm = Opdyts(evaluator,
        candidate_set_size = 16,
        number_of_transitions = 20,
        perturbation_length = 50,
        seed = 0
    )

    assert np.round(Loop(threshold = 1e-3, maximum_cost = 10000).run(
        evaluator = evaluator,
        algorithm = algorithm
    )) == 231

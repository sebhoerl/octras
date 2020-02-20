from ..cases import CongestionSimulator, CongestionProblem

from octras.algorithms import CMAES
from octras import Loop, Evaluator

import pytest
import numpy as np

def test_cma_es():
    for seed in (1000, 2000, 3000, 4000):
        evaluator = Evaluator(
            simulator = CongestionSimulator(),
            problem = CongestionProblem(0.3, iterations = 200)
        )

        algorithm = CMAES(evaluator,
            initial_step_size = 50,
            seed = seed
        )

        assert abs(np.round(Loop(threshold = 1e-4).run(
            evaluator = evaluator,
            algorithm = algorithm
        )) - 230) < 10

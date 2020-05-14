from ..cases import QuadraticSimulator, QuadraticProblem, SISSimulator
from ..cases import RosenbrockSimulator, RosenbrockProblem, SISProblem

from octras.algorithms import BatchBayesianOptimization
from octras import Loop, Evaluator

import pytest, warnings
import numpy as np

def __test_bayesian_optimization():
    evaluator = Evaluator(
        simulator = QuadraticSimulator(),
        problem = QuadraticProblem([2.0, 1.0])
    )

    algorithm = BatchBayesianOptimization(
        evaluator, batch_size = 4
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(0)

        assert Loop(threshold = 1e-2).run(
            evaluator = evaluator,
            algorithm = algorithm
        ) == pytest.approx((2.0, 1.0), 1e-1)

def __test_sis_single_fidelity():
    evaluator = Evaluator(
        simulator = SISSimulator(),
        problem = SISProblem(0.6)
    )

    algorithm = BatchBayesianOptimization(
        evaluator, batch_size = 4
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(0)

        assert Loop(threshold = 1e-2).run(
            evaluator = evaluator,
            algorithm = algorithm
        ) == pytest.approx(0.625, 1e-1)

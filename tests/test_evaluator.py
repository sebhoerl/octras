import pytest

from .cases import RosenbrockSimulator
from .cases import RosenbrockProblem

from octras import Evaluator

def test_rosenbrock_evaluation():
    simulator = RosenbrockSimulator()

    evaluator = Evaluator(problem = RosenbrockProblem(3), simulator = simulator)
    identifier = evaluator.submit([1, 1, 1])
    assert evaluator.get(identifier)[0] == 0.0

    evaluator = Evaluator(problem = RosenbrockProblem(5), simulator = simulator)
    identifier = evaluator.submit([-1, 1, 1, 1, 1])
    assert evaluator.get(identifier)[0] == 4.0

    evaluator = Evaluator(problem = RosenbrockProblem(4), simulator = simulator)

    identifier1 = evaluator.submit([-1, 1, 1, 1])
    assert evaluator.get(identifier1)[0] == 4.0

    identifier2 = evaluator.submit([-1, 1, 2, 1])
    assert evaluator.get(identifier2)[0] != 4.0

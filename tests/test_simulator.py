import pytest

from .cases.rosenbrock import RosenbrockSimulator
from .cases.quadratic import QuadraticSimulator
from .cases.traffic import TrafficSimulator

def test_rosenbrock_simulator():
    simulator = RosenbrockSimulator()

    simulator.run("A", dict(x = [1, 1, 1]))
    simulator.run("B", dict(x = [-1, 1, 1, 1]))
    simulator.run("C", dict(x = [-1, 1, 1, 1, 1]))
    simulator.run("D", dict(x = [-1, 1, 2, 1]))

    assert simulator.ready("A")
    assert simulator.ready("B")
    assert simulator.ready("C")
    assert simulator.ready("D")

    assert simulator.get("A") == 0.0
    assert simulator.get("B") == 4.0
    assert simulator.get("C") == 4.0
    assert simulator.get("D") != 0.0

    simulator.clean("A")
    simulator.clean("B")
    simulator.clean("C")
    simulator.clean("D")

def test_quadratic_simulator():
    simulator = QuadraticSimulator()

    simulator.run("A", dict(u = [0], x = [1]))
    simulator.run("B", dict(u = [0], x = [2]))
    simulator.run("C", dict(u = [0], x = [0]))
    simulator.run("D", dict(u = [-2], x = [-2]))

    assert simulator.ready("A")
    assert simulator.ready("B")
    assert simulator.ready("C")
    assert simulator.ready("D")

    assert simulator.get("A") == 1.0
    assert simulator.get("B") == 4.0
    assert simulator.get("C") == 0.0
    assert simulator.get("D") == 0.0

    simulator.clean("A")
    simulator.clean("B")
    simulator.clean("C")
    simulator.clean("D")

# TODO: Provide proper unit test for the test simulator
# - Based on known values
# - And one for the restarting behaviour

#def test_traffic_simulator():
#    simulator = TrafficSimulator()
#
#    simulator.run("x", dict(capacities = [510.0, 412.0], scaling = 1.0, iterations = 100))
#    assert simulator.get("x") == pytest.approx([510.0, 412.0])
#
#    simulator.run("x", dict(capacities = [200.0, 600.0], scaling = 1.0, iterations = 100))
#    assert simulator.get("x") == pytest.approx([738, 1262])
#
#    simulator.run("run0", dict(capacities = [200.0, 600.0], scaling = 1.0, iterations = 100))
#    current_run = "run0"
#    runs = 0
#
#    while simulator.get(current_run) != 0.693:
#        runs += 1
#
#        previous_run = "run%d" % (runs - 1)
#        current_run = "run%d" % runs
#
#        simulator.run(current_run, dict(capacity = 600, iterations = 10, restart = previous_run, random_seed = runs * 1000))
#
#    assert runs == 7

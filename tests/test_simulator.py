import pytest
from .cases import RosenbrockSimulator, QuadraticSimulator, SISSimulator, CongestionSimulator

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

def test_sis_simulator():
    simulator = SISSimulator()

    simulator.run("dieout1", dict(steps = 5000, dt = 1e-2, beta = 0.1, gamma = 0.4)) # 0
    simulator.run("dieout2", dict(steps = 5000, dt = 1e-2, beta = 0.1, gamma = 0.6)) # 0
    simulator.run("permanent1", dict(steps = 5000, dt = 1e-2, beta = 0.8, gamma = 0.2)) # 750
    simulator.run("permanent2", dict(steps = 5000, dt = 1e-2, beta = 0.8, gamma = 0.1)) # 875

    assert simulator.ready("dieout1")
    assert simulator.ready("dieout2")
    assert simulator.ready("permanent1")
    assert simulator.ready("permanent2")

    assert simulator.get("dieout1")[1] == pytest.approx(0.0, abs = 1e-3)
    assert simulator.get("dieout2")[1] == pytest.approx(0.0, abs = 1e-3)

    assert simulator.get("permanent1")[1] == pytest.approx(0.75, abs = 1e-3)
    assert simulator.get("permanent2")[1] == pytest.approx(0.875, abs = 1e-3)

    simulator.clean("dieout1")
    simulator.clean("dieout2")
    simulator.clean("permanent1")
    simulator.clean("permanent2")

def test_sis_simulator_restart():
    simulator = SISSimulator()

    simulator.run("run0", dict(steps = 100, dt = 1e-2, beta = 0.8, gamma = 0.2))

    current_run = "run0"
    runs = 0

    while abs(simulator.get(current_run)[1] - 0.75) > 1e-3:
        runs += 1

        previous_run = "run%d" % (runs - 1)
        current_run = "run%d" % runs

        simulator.run(current_run, dict(steps = 100, dt = 1e-2, beta = 0.8, gamma = 0.2, restart = previous_run))

    assert runs == 12

def test_congestion_simulator():
    simulator = CongestionSimulator()

    simulator.run("x", dict(capacity = 800))
    assert simulator.get("x") == pytest.approx(0.868)

    simulator.run("x", dict(capacity = 600))
    assert simulator.get("x") == pytest.approx(0.695)

    simulator.run("run0", dict(capacity = 600, iterations = 10))
    current_run = "run0"
    runs = 0

    while simulator.get(current_run) != 0.693:
        runs += 1

        previous_run = "run%d" % (runs - 1)
        current_run = "run%d" % runs

        simulator.run(current_run, dict(capacity = 600, iterations = 10, restart = previous_run, random_seed = runs * 1000))

    assert runs == 7

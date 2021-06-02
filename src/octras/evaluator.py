import uuid, time, logging, deep_merge

from .simulator import Simulator
from .problem import Problem

logger = logging.getLogger("octras")

class Evaluator:
    def __init__(self, problem : Problem, simulator : Simulator, interval = 0.0, parallel = 1, follow_trace = True):
        self.problem = problem
        self.simulator = simulator
        self.interval = interval
        self.parallel = parallel

        self.simulations = {}

        self.pending = []
        self.running = []
        self.finished = []

        self.current_evaluations = 0
        self.current_cost = 0

        self.follow_trace = follow_trace
        self.trace = []

        problem_information = problem.get_information()

        if not"number_of_parameters" in problem_information:
            raise RuntimeError("Problem information does not provide number_of_parameters.")

        self.number_of_parameters = problem_information["number_of_parameters"]

    def _create_identifier(self):
        identifier = str(uuid.uuid4())

        while identifier in self.simulations:
            identifier = str(uuid.uuid4())

        return identifier

    def submit(self, x, simulator_parameters = {}, annotations = {}, transient = False):
        if len(x) != self.number_of_parameters:
            raise RuntimeError("Invalid number of parameters: %d (expected %d)" % (
                len(x), self.number_of_parameters
            ))

        identifier = self._create_identifier()
        response = self.problem.parameterize(x)

        if isinstance(response, tuple):
            parameters, cost = response
        else:
            parameters, cost = response, 1

        parameters = deep_merge.merge(parameters, simulator_parameters)

        self.simulations[identifier] = {
            "identifier": identifier,
            "parameters": parameters, "x": x,
            "cost": cost, "annotations": annotations,
            "status": "pending", "transient": transient
        }

        self.pending.append(identifier)
        return identifier

    def _ping(self):
        for identifier in self.running:
            if self.simulator.ready(identifier):
                simulation = self.simulations[identifier]

                simulation["result"] = self.simulator.get(identifier)
                response = self.problem.evaluate(simulation["x"], simulation["result"])

                information = None
                state = None

                if isinstance(response, tuple):
                    objective = response[0]

                    if len(response) > 1:
                        state = response[1]

                    if len(response) > 2:
                        information = response[2]
                else:
                    objective = response

                if not state is None:
                    problem_information = self.problem.get_information()

                    if not "number_of_states" in problem_information:
                        raise RuntimeError("Problem return state, but problem information does not provide number_of_states")

                    number_of_states = problem_information["number_of_states"]

                    if not len(state) == number_of_states:
                        raise RuntimeError("Wrong number of states provided: %d (expected %d)" % (
                            len(state), number_of_states
                        ))

                self.current_evaluations += 1
                self.current_cost += simulation["cost"]

                simulation["objective"] = objective
                simulation["state"] = state
                simulation["status"] = "finished"
                simulation["information"] = information

                simulation["evaluator_evaluations"] = self.current_evaluations
                simulation["evaluator_cost"] = self.current_cost

                self.running.remove(identifier)
                self.finished.append(identifier)

                if self.follow_trace:
                    self.trace.append(simulation)

        while len(self.running) < self.parallel and len(self.pending) > 0:
            simulation = self.simulations[self.pending.pop(0)]
            simulation["status"] = "running"

            self.simulator.run(simulation["identifier"], simulation["parameters"])
            self.running.append(simulation["identifier"])

    def wait(self, identifiers = None):
        if identifiers is None:
            identifiers = [
                identifier for identifier, simulation in self.simulations.items()
                if simulation["status"] != "finished"
            ]

        if isinstance(identifiers, str):
            identifiers = [identifiers]

        waiting = set(identifiers)

        initial_count = len(waiting)
        current_count = 0

        while len(waiting) > 0:
            self._ping()

            for identifier in set(waiting):
                if self.simulations[identifier]["status"] == "finished":
                    waiting.remove(identifier)

            if current_count != len(waiting):
                current_count = len(waiting)
                logger.info("Waiting for samples. %d/%d finished ..." % (initial_count - current_count, initial_count))

            if len(waiting) > 0:
                time.sleep(self.interval)

    def get(self, identifiers):
        if isinstance(identifiers, str):
            self.wait([identifiers])
            return self.simulations[identifiers]["objective"], self.simulations[identifiers]["state"]

        else:
            self.wait(identifiers)

            return [
                (self.simulations[identifier]["objective"], self.simulations[identifier]["state"])
                for identifier in identifiers
            ]

    def ready(self, identifier):
        self._ping()
        return self.simulations[identifier]["status"] == "finished"

    def clean(self, identifiers = None):
        if identifiers is None:
            identifiers = self.finished[:]
        elif isinstance(identifiers, str):
            identifiers = [identifiers]

        self.wait(identifiers)

        for identifier in identifiers:
            del self.simulations[identifier]
            self.simulator.clean(identifier)
            self.finished.remove(identifier)

    def fetch_trace(self):
        trace, self.trace = self.trace[:], []
        return trace

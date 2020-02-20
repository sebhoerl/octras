import uuid, time, logging, deep_merge

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, problem, simulator, interval = 1.0, parallel = 1):
        self.problem = problem
        self.simulator = simulator
        self.interval = interval
        self.parallel = parallel

        self.simulations = {}

        self.pending = []
        self.running = []
        self.finished = []

        self.current_runs = 0
        self.current_cost = 0

        self.trace = []

    def _create_identifier(self):
        identifier = str(uuid.uuid4())

        while identifier in self.simulations:
            identifier = str(uuid.uuid4())

        return identifier

    def submit(self, x, simulator_parameters = {}, annotations = {}, transient = False):
        if len(x) != self.problem.number_of_parameters:
            raise RuntimeError("Invalid number of parameters: %d (expected %d)" % (
                len(x), self.probem.number_of_parameters
            ))

        identifier = self._create_identifier()
        response = self.problem.prepare(x)

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

                result = self.simulator.get(identifier)
                response = self.problem.evaluate(simulation["x"], result)

                if isinstance(response, tuple):
                    objective, state = response
                else:
                    objective, state = response, None

                if not state is None:
                    if not len(state) == self.problem.number_of_states:
                        raise RuntimeError("Wrong number of states provided: %d (expected %d)" % (
                            len(state), self.problem.number_of_states
                        ))

                self.current_runs += 1
                self.current_cost += simulation["cost"]

                simulation["objective"] = objective
                simulation["state"] = state
                simulation["status"] = "finished"

                simulation["evaluator_runs"] = self.current_runs
                simulation["evaluator_cost"] = self.current_cost

                self.running.remove(identifier)
                self.finished.append(identifier)

                self.trace.append(simulation)

        while len(self.running) < self.parallel and len(self.pending) > 0:
            simulation = self.simulations[self.pending.pop(0)]
            simulation["status"] = "running"

            self.simulator.run(simulation["identifier"], simulation["parameters"])
            self.running.append(simulation["identifier"])

    def wait(self, identifiers = None):
        if identifiers is None:
            identifiers = [
                identifier for identifier, simulation in self.simulation.items()
                if simulation["status"] == "running"
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
        if isinstance(identifier, str):
            self.wait([identifier])
            return self.simulations[identifier]["objective"], self.simulations[identifier]["state"]

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

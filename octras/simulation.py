import uuid, time

class Simulator:
    def run(self, identifier, parameters):
        raise RuntimeError()

    def is_running(self, identifier):
        raise RuntimeError()

    def get_result(self, identifier):
        raise RuntimeError()

    def get_progress(self, identifier):
        return 0, 0

    def cleanup(self, identifier):
        pass

class Scheduler:
    def __init__(self, simulator, default_parameters = {}, number_of_runners = 1, ping_time = 1.0):
        self.number_of_runners = number_of_runners
        self.simulator = simulator
        self.default_parameters = default_parameters
        self.ping_time = ping_time

        self.registry = {}

        self.pending = []
        self.running = set()
        self.finished = set()

    def _create_identifier(self):
        new_id = uuid.uuid4()

        while new_id in self.registry:
            new_id = uuid.uuid4()

        return new_id

    def schedule(self, simulation_parameters):
        identifier = self._create_identifier()

        parameters = {}
        parameters.update(self.default_parameters)
        parameters.update(simulation_parameters)

        self.registry[identifier] = parameters
        self.pending.append(identifier)

        return identifier

    def _ping(self):
        finished_identifiers = set()

        for identifier in self.running:
            if not self.simulator.is_running(identifier):
                finished_identifiers.add(identifier)

        for identifier in finished_identifiers:
            self.running.remove(identifier)
            self.finished.add(identifier)

        while len(self.running) < self.number_of_runners and len(self.pending) > 0:
            identifier = self.pending.pop(0)
            self.simulator.run(identifier, self.registry[identifier])
            self.running.add(identifier)

    def wait(self, identifiers = None, verbose = True):
        if identifiers is None:
            identifiers = set(self.pending) | set(self.running)
        else:
            identifiers = set(identifiers)

        previous_progress = None

        while len(identifiers - self.finished):
            self._ping()
            time.sleep(self.ping_time)

            if verbose:
                current, total = 0, 0

                for identifier in identifiers:
                    local_current, local_total = self.simulator.get_progress(identifier)
                    current += local_current
                    total += local_total

                if current != previous_progress and total > 0:
                    print("Progress: %d / %d" % (current, total))
                    previous_progress = current


    def get(self, identifier, verbose = False):
        self.wait([identifier], verbose = verbose)

        return {
            "result": self.simulator.get_result(identifier),
            "parameters": self.registry[identifier]
        }

    def cleanup(self, identifier):
        self.simulator.cleanup(identifier)

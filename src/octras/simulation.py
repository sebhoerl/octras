import uuid, time, logging
logger = logging.getLogger(__name__)

class Simulator:
    def run(self, identifier, parameters):
        raise RuntimeError()

    def is_running(self, identifier):
        raise RuntimeError()

    def get_result(self, identifier):
        raise RuntimeError()

    def get_cost(self, identifier):
        return 1.0

    def cleanup(self, identifier):
        pass

class Scheduler:
    def __init__(self, simulator, number_of_runners = 1, ping_time = 1.0):
        self.number_of_runners = number_of_runners
        self.simulator = simulator
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
        parameters.update(simulation_parameters)

        self.registry[identifier] = parameters
        self.pending.append(identifier)

        logger.info("Scheduled simulation %s." % identifier)

        return identifier

    def _ping(self):
        finished_identifiers = set()

        for identifier in self.running:
            if not self.simulator.is_running(identifier):
                finished_identifiers.add(identifier)

        for identifier in finished_identifiers:
            self.running.remove(identifier)
            self.finished.add(identifier)
            logger.info("Finished simulation %s." % identifier)

        while len(self.running) < self.number_of_runners and len(self.pending) > 0:
            identifier = self.pending.pop(0)
            self.simulator.run(identifier, self.registry[identifier])
            self.running.add(identifier)
            logger.info("Starting simulation %s." % identifier)

    def wait(self, identifiers = None):
        if identifiers is None:
            identifiers = set(self.pending) | set(self.running)
        else:
            identifiers = set(identifiers)

        previous_count = None

        while len(identifiers - self.finished) > 0:
            self._ping()
            time.sleep(self.ping_time)

            total_count = len(identifiers)
            remaining_count = len(identifiers - self.finished)
            finished_count = total_count - remaining_count

            if finished_count != previous_count and finished_count < total_count:
                previous_count = finished_count
                logger.info("Waiting for simulations (%d/%d) ..." % (finished_count + 1, total_count))

    def get_result(self, identifier):
        self.wait([identifier])
        return self.simulator.get_result(identifier)

    def get_cost(self, identifier):
        self.wait([identifier])
        return self.simulator.get_cost(identifier)

    def cleanup(self, identifier):
        self.simulator.cleanup(identifier)
        logger.info("Cleaned up simulation %s." % identifier)

import numpy as np
import logging

logger = logging.getLogger(__name__)

class Loop:
    def __init__(self, maximum_cost = np.inf, maximum_runs = np.inf, threshold = 0.0):
        self.maximum_cost = maximum_cost
        self.maximum_runs = maximum_runs
        self.threshold = threshold

        self.objective = None
        self.x = None

    def _process(self, simulation):
        self.cost = simulation["evaluator_cost"]
        self.runs = simulation["evaluator_runs"]

        if self.objective is None or simulation["objective"] < self.objective:
            if not ("transient" in simulation and simulation["transient"]):
                self.objective = simulation["objective"]
                self.x = simulation["x"]

                logger.info("Found new best objective (%f) at %s" % (
                    self.objective, str(simulation["x"])
                ))

    def run(self, evaluator, algorithm):
        while True:
            if evaluator.current_cost > self.maximum_cost:
                logger.warn("Stopping because of cost limit is reached.")
                break

            if evaluator.current_runs > self.maximum_runs:
                logger.warn("Stopping because of run limit is reached.")
                break

            if not self.objective is None and self.objective < self.threshold:
                logger.info("Stopping because of objective is minized.")
                break

            algorithm.advance()
            map(self._process, evaluator.fetch_trace())

            if not self.objective is None:
                logger.info("Best objective found: %f" % self.objective)
                logger.info("  at %s" % str(self.x))

        return self.x

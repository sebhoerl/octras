import numpy as np
import logging

logger = logging.getLogger("octras")

class Loop:
    def __init__(self, maximum_cost = np.inf, maximum_evaluations = np.inf, threshold = 0.0):
        self.maximum_cost = maximum_cost
        self.maximum_evaluations = maximum_evaluations
        self.threshold = threshold

        self.objective = None
        self.x = None

    def _process(self, simulation):
        self.cost = simulation["evaluator_cost"]
        self.evaluations = simulation["evaluator_evaluations"]

        if self.objective is None or simulation["objective"] < self.objective:
            if not ("transient" in simulation and simulation["transient"]):
                self.objective = simulation["objective"]
                self.x = simulation["x"]

                logger.info("Found new best objective (%f) at %s" % (
                    self.objective, str(simulation["x"])
                ))

    def run(self, evaluator, algorithm, tracker = None):
        initial_evaluations = evaluator.current_evaluations
        initial_cost = evaluator.current_cost

        while True:
            if evaluator.current_cost - initial_cost > self.maximum_cost:
                logger.warn("Stopping because of cost limit is reached.")
                break

            if evaluator.current_evaluations - initial_evaluations > self.maximum_evaluations:
                logger.warn("Stopping because of run limit is reached.")
                break

            if not self.objective is None and self.objective < self.threshold:
                logger.info("Stopping because of objective is minized.")
                break

            algorithm.advance(evaluator)

            trace = evaluator.fetch_trace()

            for item in trace:
                self._process(item)

                if not tracker is None:
                    tracker.notify(item)

            if not self.objective is None:
                logger.info("Best objective found: %f" % self.objective)
                logger.info("  at %s" % str(self.x))

        return self.x

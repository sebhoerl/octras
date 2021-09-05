import logging
import pickle

logger = logging.getLogger("octras")

class PickleTracker:
    def __init__(self, output_path):
        self.output_path = output_path
        self.history = []

        self.best_objective = None

    def notify(self, simulation):
        if self.best_objective is None or simulation["objective"] < self.best_objective:
            self.best_objective = simulation["objective"]
            logger.info("New best objective: %f" % self.best_objective)

        self.history.append(simulation)

        with open(self.output_path, "wb+") as f:
            pickle.dump(self.history, f)

logger = logging.getLogger("octras")

class LogTracker:
    def __init__(self):
        self.best_objective = None

    def notify(self, simulation):
        if self.best_objective is None or simulation["objective"] < self.best_objective:
            if not ("transient" in simulation and simulation["transient"]):
                self.best_objective = simulation["objective"]

                logger.info("Found new best objective (%f) at %s" % (
                    self.best_objective, str(simulation["x"])
                ))

class CompositeTracker:
    def __init__(self, trackers):
        self.trackers = trackers

    def notify(self, simulation):
        for tracker in self.trackers:
            tracker.notift(simulation)

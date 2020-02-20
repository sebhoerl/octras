import numpy as np

class Tracker:
    def __init__(self, maximum_cost = np.inf, maximum_runs = np.inf):
        self.maximum_cost = maximum_cost
        self.maximum_runs = maximum_runs

    def notify(self, simulation):
        pass

    

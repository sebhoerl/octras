import matplotlib.pyplot as plt

class PlotTracker:
    def __init__(self):
        self.best_objectives = []
        self.objectives = []

        self.best_objective = None

    def notify(self, simulation):
        self.objectives.append(simulation["objective"])

        if self.best_objective is None or simulation["objective"] < self.best_objective:
            self.best_objective = simulation["objective"]

        self.best_objectives.append(self.best_objective)

    def show(self):
        plt.figure()
        plt.plot(self.best_objectives)
        plt.plot(self.objectives, '.')
        plt.title(self.best_objective)
        plt.show()

from octras import Problem

import pandas as pd
import numpy as np
import os.path

NAMES = [
    "majorFactor",
    "immediateFactor",
    "minorFactor",
]

BOUNDS = [
    (0.25, 1.25),
    (0.25, 1.25),
    (0.25, 1.25),
]

INITIALS = [
    1.0,
    1.0,
    1.0
]

class ParisFlowProblem(Problem):
    """
        Here we define an optimization problem for MATSim. Namely, we want to
        push the mode shares of car and public transport to each 30% by playing
        around with the mode constants.
    """

    def __init__(self, analyzer, sampling_rate, threads, iterations, config_path, reference_path):
        """
            We pass some desired reference values and save them.
        """

        # We need to provide some mandatory information
        self.number_of_parameters = 3
        self.number_of_states = 1

        # We can provide some information that is passed to the output
        self.information = dict(problem = "ParisProblem")

        # We can provide the reference state. Currently, it is not used in any
        # optimizer, but it is helpful for plotting optimization results as
        # the existing plotting scripts will know how to use this information.
        self.reference_state = 0

        self.initial = INITIALS
        self.bounds = BOUNDS

        self.analyzer = analyzer
        self.threads = threads
        self.iterations = iterations
        self.config_path = config_path

        self.sampling_rate = sampling_rate
        self.reference_path = reference_path

    def prepare(self, x):
        """
            This function gets a vector of numeric parameters. Our task is to
            translate those numbers into actual config instructions for
            MATSim.
        """

        # First we need to pass some parameters to the simulator
        parameters = dict(config = { # We provide --config: parameters to the simulation
            "qsim.numberOfThreads": self.threads,
            "global.numberOfThreads": self.threads,
            "eqasim.tripAnalysisInterval": self.iterations,
            "controler.writeTripsInterval": 0,
            "linkStats.writeLinkStatsInterval": 0,
            "counts.writeCountsInterval": 0
        }, iterations = self.iterations)

        arguments = []

        for value, name in zip(x, NAMES):
            arguments.append("--capacity:%s" % name)

            value = np.maximum(0.25, value)
            value = np.minimum(1.75, value)

            arguments.append(str(value * self.sampling_rate))

        arguments += [
            "--config-path", self.config_path,
            "--use-epsilon",
            "--convergence-threshold", "0.05",
            "--flow-path", os.path.realpath(self.reference_path)
        ]

        parameters["arguments"] = arguments

        return parameters

    def evaluate(self, x, path):
        """
            This is the objective function. It receives the numeric parameters
            as the first argument and the output of the simulator as the
            second argument. In this case, we rely on the fact that the
            standard MATSim simulator just returns the path of the output
            directory of the simulation.
        """

        result = self.analyzer.execute(path)
        return result["objective"], None, result

from octras import Problem

import pandas as pd
import numpy as np
import os.path

NAMES = [
    "capacity:major",
    "capacity:intermediate",
    "capacity:minor",
    "capacity:link",
    "config:eqasim.crossingPenalty"
    "mode-choice-parameter:car.alpha_u",
    "mode-choice-parameter:car.betaTravelTime_u_min",
]

BOUNDS = [
    (0.25, 1.75),
    (0.25, 1.75),
    (0.25, 1.75),
    (0.25, 1.75),
    (0.0, 10.0),
    (-3.0, 3.0),
    (-3.0, 0.0)
]

INITIALS = [
    1.0,
    1.0,
    1.0,
    1.0,
    3.0,
    1.35,
    -0.06
]

class ParisFlowProblem(Problem):
    """
        Here we define an optimization problem for MATSim. Namely, we want to
        push the mode shares of car and public transport to each 30% by playing
        around with the mode constants.
    """

    def __init__(self, analyzer, threads, iterations, config_path, reference_path):
        """
            We pass some desired reference values and save them.
        """

        # We need to provide some mandatory information
        self.number_of_parameters = 7
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

        for index, (value, name) in enumerate(zip(x, NAMES)):
            arguments.append("--%s" % name)

            value = np.maximum(BOUNDS[index][0], value)
            value = np.minimum(BOUNDS[index][1], value)

            arguments.append(str(value))

        arguments += [
            "--config-path", self.config_path,
            "--use-epsilon", # Temporarily removed to keep modes fix
            #"--fix-modes", # To keep modes fix
            "--convergence-threshold", "0.05",
            "--flow-path", os.path.realpath(self.reference_path),
            "--config:qsim.storageCapacityFactor", "1000000"
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

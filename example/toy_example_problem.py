from octras import Problem
import pandas as pd
import numpy as np
import glob

class ToyExampleProblem(Problem):
    """
        Here we define an optimization problem for MATSim. Namely, we want to
        push the mode shares of car and public transport to each 30% by playing
        around with the mode constants.
    """

    def __init__(self, reference_share):
        """
            We pass some desired reference values and save them.
        """

        self.reference_share = reference_share

        # We need to provide some mandatory information
        self.number_of_parameters = 1
        self.number_of_states = 1

        # We can provide some information that is passed to the output
        self.information = dict(problem = "ToyExampleProblem")

        # We can provide the reference state. Currently, it is not used in any
        # optimizer, but it is helpful for plotting optimization results as
        # the existing plotting scripts will know how to use this information.
        self.reference_state = [self.reference_share]

        self.initial = [0.5]
        self.bounds = [(0.01, 1.0)]

    def prepare(self, x):
        """
            This function gets a vector of numeric parameters. Our task is to
            translate those numbers into actual config instructions for
            MATSim.
        """

        # First we need to pass some parameters to the simulator
        parameters = dict(config = { # We provide --config: parameters to the simulation
            "qsim.flowCapacityFactor": max(0.01, min(1.0, x[0])), # Here we can pass config options
        })

        return parameters

    def evaluate(self, x, path):
        """
            This is the objective function. It receives the numeric parameters
            as the first argument and the output of the simulator as the
            second argument. In this case, we rely on the fact that the
            standard MATSim simulator just returns the path of the output
            directory of the simulation.
        """

        # Read the modestats file as CSV
        mode_stats_paths = glob.glob("%s/*modestats.txt" % path)
        df = pd.read_csv(mode_stats_paths[0], sep = "\t")

        simulation_share = df["pt"].values[-1] # Share of pt trips

        # We construct a vector holding the *state* of the simulation. This is not
        # used by most simulators, but important, for instance, for Opdyts!
        state = [simulation_share]

        # Here we construct an objective value. Here, we want to minimize
        # the quadratic error between the observed and reference mode shares for
        # car and public transport.
        objective = np.abs(simulation_share - self.reference_share)

        # Return state and objective
        return objective, state

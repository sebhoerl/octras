import numpy as np
import numpy.linalg as la

import logging
logger = logging.getLogger("octras")

from octras import Evaluator

# https://en.wikipedia.org/wiki/CMA-ES

class CMAES:
    def __init__(self, problem, candidate_set_size = None, initial_step_size = 0.3, seed = None):
        problem_information = problem.get_information()

        if not "initial_values" in problem_information:
            raise RuntimeError("CMA-ES expects initial_values in problem information.")

        if not "number_of_parameters" in problem_information:
            raise RuntimeError("CMA-ES expects number_of_parameters in problem information.")

        number_of_parameters = problem_information["number_of_parameters"]
        self.initial_values = problem_information["initial_values"]

        # Selection parameters
        L_default = 4 + int(np.floor(3 * np.log(number_of_parameters)))
        self.L = L_default if candidate_set_size is None else candidate_set_size

        if not candidate_set_size is None and candidate_set_size < L_default:
            logger.warning("Using requested candidate set size %d (recommended is at least %d!)" % (candidate_set_size, L_default))

        # Initialize static parameters
        self.N = number_of_parameters

        self.mu = self.L / 2.0
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.mu = int(np.floor(self.mu))
        self.weights = self.weights[:self.mu] / np.sum(self.weights[:self.mu])
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights ** 2)

        # Adaptation parameters
        self.cc = (4 + self.mueff / self.N) / (self.N + 4 + 2.0 * self.mueff / self.N)
        self.cs = (self.mueff + 2.0) / (self.N + self.mueff + 5.0)
        self.c1 = 2.0 / ((self.N + 1.3)**2 + self.mueff)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((self.N + 2.0)**2 + self.mueff))
        self.damps = 1.0 + 2.0 * max(0, np.sqrt((self.mueff - 1.0) / (self.N + 1.0)) - 1.0) + self.cs

        # Initialize dynamic parameters
        self.pc = np.zeros((self.N,1))
        self.ps = np.zeros((self.N,1))
        self.B = np.eye(self.N)
        self.D = np.ones((self.N,))
        self.C = np.dot(self.B, np.dot(np.diag(self.D**2), self.B.T))
        self.invsqrtC = np.dot(self.B, np.dot(np.diag(self.D**-1), self.B.T))
        self.eigeneval = 0
        self.counteval = 0
        self.chiN = self.N**0.5 * (1.0 - 1.0 / (4.0 * self.N) + 1.0 / (21.0 * self.N**2))

        # Initialize algorithm parameters
        self.iteration = 0
        self.mean = None
        self.sigma = initial_step_size

        self.random = np.random.RandomState(seed)

    def advance(self, evaluator: Evaluator):
        if self.iteration == 0:
            self.mean = np.copy(self.initial_values).reshape((self.N, 1))

        self.iteration += 1
        logger.info("Starting CMA-ES iteration %d." % self.iteration)

        annotations = {
            "mean": self.mean,
            "covariance": self.C, "pc": self.pc, "ps": self.ps,
            "sigma": self.sigma
        }

        self.counteval += self.L

        candidate_parameters = self.sigma * np.dot(
            (self.random.normal(size = (self.N, self.L)) * self.D[:, np.newaxis]).T, self.B
        ) + self.mean.T

        candidate_identifiers = [
            evaluator.submit(parameters, annotations = annotations)
            for parameters in candidate_parameters
        ]

        # Wait for samples
        evaluator.wait()

        # Obtain fitness
        candidate_objectives = np.array([
            evaluator.get(identifier)[0] # We minimize!
            for identifier in candidate_identifiers
        ])

        # Cleanup
        for identifier in candidate_identifiers:
            evaluator.clean(identifier)

        sorter = np.argsort(candidate_objectives)

        candidate_objectives = candidate_objectives[sorter]
        candidate_parameters = candidate_parameters[sorter, :]

        # Update mean
        previous_mean = self.mean
        self.mean = np.sum(candidate_parameters[:self.mu] * self.weights[:, np.newaxis], axis = 0).reshape((self.N, 1))

        # Update evolution paths
        psa = (1.0 - self.cs ) * self.ps
        psb = np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * np.dot(self.invsqrtC, self.mean - previous_mean) / self.sigma
        self.ps = psa + psb

        hsig = la.norm(self.ps) / np.sqrt(1.0 - (1.0 - self.cs)**(2.0 * self.counteval / self.L)) / self.chiN < 1.4 + 2.0 / (self.N + 1.0)
        pca = (1.0 - self.cc) * self.pc
        pcb = hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * (self.mean - previous_mean) / self.sigma
        self.pc = pca + pcb

        # Adapt covariance matrix
        artmp = (1.0 / self.sigma) * (candidate_parameters[:self.mu].T - previous_mean)

        Ca = (1.0 - self.c1 - self.cmu) * self.C
        Cb = self.c1 * (np.dot(self.pc, self.pc.T) + (not hsig) * self.cc * (2.0 - self.cc) * self.C)
        Cc = self.cmu * np.dot(artmp, np.dot(np.diag(self.weights), artmp.T))
        C = Ca + Cb + Cc

        # Adapt step size
        self.sigma = self.sigma * np.exp((self.cs / self.damps) * (la.norm(self.ps) / self.chiN - 1.0))

        if self.counteval - self.eigeneval > self.L / (self.c1 + self.cmu) / self.N / 10.0:
            self.eigeneval = self.counteval

            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            d, self.B = la.eig(self.C)

            self.D = np.sqrt(d)
            Dm = np.diag(1.0 / np.sqrt(d))

            self.invsqrtC = np.dot(self.B.T, np.dot(Dm, self.B))

        if np.max(self.D) > 1e7 * np.min(self.D):
            logger.warning("Condition exceeds 1e14")

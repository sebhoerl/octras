from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.initial_designs import LatinDesign

from emukit.bayesian_optimization.acquisitions import NegativeLowerConfidenceBound
from emukit.bayesian_optimization.loops.bayesian_optimization_loop import BayesianOptimizationLoop

from emukit.core.loop import UserFunctionWrapper

import GPy
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper

import numpy as np

import logging
logger = logging.getLogger(__name__)

class BatchBayesianOptimization:
    def __init__(self, evaluator, batch_size, initial_sample_count = None, number_of_restarts = 1):
        self.evaluator = evaluator
        self.problem = evaluator.problem

        self.batch_size = batch_size

        self.initial_sample_count = batch_size if initial_sample_count is None else initial_sample_count
        self.number_of_restarts = number_of_restarts

        if not hasattr(self.problem, "bounds"):
            raise RuntimeError("Problem needs to provide bounds if BatchBayesianOptimization is used.")

        self.iteration = 0
        self.loop = None

    def _evaluate_batch(self, X):
        identifiers = [
            self.evaluator.submit(parameters)
            for parameters in X
        ]

        response = [
            self.evaluator.get(identifier)[0]
            for identifier in identifiers
        ]

        for identifier in identifiers:
            self.evaluator.clean(identifier)

        return np.array(response).reshape((-1, 1))

    def initialize(self):
        parameter_space = ParameterSpace([
            ContinuousParameter("x%d" % index, bounds[0], bounds[1])
            for index, bounds in enumerate(self.problem.bounds)
        ])

        # Obtain initial sample
        design = LatinDesign(parameter_space)
        initial_parameters = design.get_samples(self.initial_sample_count)
        initial_response = self._evaluate_batch(initial_parameters)

        kernel = None # GPy.kern.RBF(1)
        model = GPy.models.GPRegression(initial_parameters, initial_response, kernel)
        model = GPyModelWrapper(model)

        acquisition = NegativeLowerConfidenceBound(model)

        self.loop = BayesianOptimizationLoop(
            model = model, space = parameter_space,
            acquisition = acquisition, batch_size = self.batch_size
        )

    def advance(self):
        if self.iteration == 0:
            logger.info("Initializing Batch Bayesian Optimization")
            self.initialize()

        self.iteration += 1
        logger.info("Starting BOO iteration %d" % self.iteration)

        self.loop.run_loop(UserFunctionWrapper(self._evaluate_batch), 1)


class MultiFidelityBatchBayesianOptimization:
    def __init__(self, evaluator, batch_size, initial_sample_count = None, number_of_restarts = 1):
        self.evaluator = evaluator
        self.problem = evaluator.problem

        self.batch_size = batch_size

        self.initial_sample_count = batch_size if initial_sample_count is None else initial_sample_count
        self.number_of_restarts = number_of_restarts

        if not hasattr(self.problem, "bounds"):
            raise RuntimeError("Problem needs to provide bounds if MultiFidelityBatchBayesianOptimization is used.")

        if not hasattr(self.problem, "fidelities"):
            raise RuntimeError("Problem needs to provide fidelities if MultiFidelityBatchBayesianOptimization is used.")

        self.iteration = 0
        self.loop = None

    def _evaluate_batch(self, X):
        identifiers = [
            self.evaluator.submit(
                parameters[:-1],
                dict(fidelity = self.problem.fidelities[int(parameters[-1])][0])
            ) for parameters in X
        ]

        response = [
            self.evaluator.get(identifier)[0]
            for identifier in identifiers
        ]

        for identifier in identifiers:
            self.evaluator.clean(identifier)

        return np.array(response).reshape((-1, 1))

    def initialize(self):
        parameter_space = ParameterSpace([
            ContinuousParameter("x%d" % index, bounds[0], bounds[1])
            for index, bounds in enumerate(self.problem.bounds)
        ] + [InformationSourceParameter(len(self.problem.fidelities))])

        # Obtain initial sample
        design = LatinDesign(parameter_space)
        initial_parameters = design.get_samples(self.initial_sample_count)
        initial_response = self._evaluate_batch(initial_parameters)

        kernels = [GPy.kern.RBF(1)] * len(self.problem.fidelities)
        kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)

        model = GPyLinearMultiFidelityModel(
            initial_parameters, initial_response,
            kernel, n_fidelities = len(self.problem.fidelities)
        )

        model = GPyMultiOutputWrapper(model, len(self.problem.fidelities))
        acquisition = NegativeLowerConfidenceBound(model)

        self.loop = BayesianOptimizationLoop(
            model = model, space = parameter_space,
            acquisition = acquisition, batch_size = self.batch_size
        )

    def advance(self):
        if self.iteration == 0:
            logger.info("Initializing Batch Bayesian Optimization")
            self.initialize()

        self.iteration += 1
        logger.info("Starting BOO iterprint(self.iteration)ation %d" % self.iteration)

        self.loop.run_loop(UserFunctionWrapper(self._evaluate_batch), 1)

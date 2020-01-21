from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter

from emukit.multi_fidelity.models.linear_model import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel import LinearMultiFidelityKernel

from emukit.model_wrappers import GPyModelWrapper

from emukit.bayesian_optimization.acquisitions import NegativeLowerConfidenceBound
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.bayesian_optimization.loops.bayesian_optimization_loop import BayesianOptimizationLoop
from emukit.core.loop.candidate_point_calculators import GreedyBatchPointCalculator
from emukit.core.initial_designs import RandomDesign

from emukit.core.loop.stopping_conditions import StoppingCondition
from emukit.bayesian_optimization.acquisitions.minvalue_entropy_search import MinValueEntropySearch, Cost, MultiFidelityMinValueEntropySearch

from emukit.model_wrappers import GPyMultiOutputWrapper
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer


import GPy
import numpy as np


class FidelityEvaluator:
    """
        A generalization over evaluator to get evaluations for different fidelity levels.
        Callable class to get evaluations and pass them to bo procedure (emukit based).
    """
    def __init__(self, evaluator, fidelities=None):
        """

        :param evaluator:
        :param fidelities: required for evaluation.
            If None, single fidelity setting is used.
            Example: TODO
        """
        self.evaluator = evaluator
        self.fidelities = fidelities
        self.annotations = {"bo_iteration": 0}

    def __call__(self, parameters):
        annotations = {}
        annotations.update(self.annotations)

        identifiers = []

        if self.fidelities is None:
            for p in parameters:
                identifiers.append(self.evaluator.schedule(p, annotations=annotations))

        else:
            for p in parameters:
                fidelity = self.fidelities[int(p[-1])]

                annotations.update({"fidelity_level": fidelity["name"]})
                fidelity_parameters = fidelity["parameters"]

                identifiers.append(self.evaluator.schedule(p[:-1], fidelity_parameters, annotations = annotations))

        self.evaluator.wait(identifiers)
        objectives = [self.evaluator.get(identifier)[0] for identifier in identifiers]

        for identifier in identifiers:
            self.evaluator.cleanup(identifier)

        return np.array(objectives).reshape((parameters.shape[0], -1))

    def update(self, loop, state):
        # @Anastasia: Here one could add additional information that is saved
        # in the output log file. Like things to display the acquisition function
        # etc... This function here is called as a iteration ends callback in
        # the BO loop.

        self.annotations["bo_iterations"] = state.iteration


class EvaluatorStoppingCondition(StoppingCondition):
    """
        Stopping condition defined in the setting of the evaluator by maximum
        cost or maximum number of iterations.
    """
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def should_stop(self, state):
        return self.evaluator.finished


def define_GPmodel_sf(use_standard_kernels, number_of_parameters, initial_x, initial_y, num_restarts=1):
    kernel = None
    if not use_standard_kernels:
        # @ Anastasia: Here convergence is very poor if I use this kernel from the notebook. Did I do something wrong?
        kernel = GPy.kern.sde_Matern52(number_of_parameters, lengthscale=1.0, variance=1.0, ARD=True)
        kernel.variance.constrain_bounded(0.05, 0.1)
        kernel.lengthscale.constrain_bounded(0, 0.1)

    model = GPy.models.GPRegression(initial_x, initial_y, kernel)
    model.Gaussian_noise.variance.fix(1e-3)

    model = GPyModelWrapper(model, n_restarts=num_restarts)
    model.optimize()

    return model


def define_GPmodel_mf(number_of_parameters, initial_x, initial_y, fidelities, num_restarts=1):
    kernel_low = GPy.kern.sde_Matern32(number_of_parameters, lengthscale=1.0, variance=0.2, ARD=True)
    kernel_low.variance.constrain_bounded(0.05, 0.1)
    kernel_low.lengthscale.constrain_bounded(0, 0.1)

    kernel_error = GPy.kern.sde_Matern52(number_of_parameters, lengthscale=1.0, variance=0.2, ARD=True)
    kernel_error.lengthscale.constrain_bounded(0, 0.1)
    kernel_error.variance.constrain_bounded(10e-5, 5.0 * 10e-5)

    kernel = LinearMultiFidelityKernel([kernel_low, kernel_error])
    kernel.constrain_bounded(0.0, 1.0, 1.0)
    kernel.scale.constrain_bounded(0.0, 0.6)

    model = GPyLinearMultiFidelityModel(initial_x, initial_y, kernel, len(fidelities))
    model.likelihood.Gaussian_noise.fix(5.0 * 1e-3)
    model.likelihood.Gaussian_noise_1.fix(1e-3)

    model = GPyMultiOutputWrapper(
        model, n_outputs=len(fidelities),
        n_optimization_restarts=num_restarts, verbose_optimization=False
        )

    model.optimize()

    return model

"""
This is the implementation of a Bayesion Optimization-based optimizer.

There are three methods:
- MES and GPBUCB for single-fidelity optimization
- MFMES for multi-fidelity optimization

For MFMES fidelities should be provided (see the respective unit test) in
  tests/test_bo -> test_multi_fidelity

@Anastasia
Currently, the use_standard_kernels flag defines whether to use the standard
kernels or the ones provided by you. I saw better convergence with the standard
ones, but maybe I did something wrong (see below)!
"""


def bo_algorithm(evaluator, batch_size=4, num_restarts=1, update_interval=1, initial_samples=4, method="mes",
                 fidelities=None, use_standard_kernels=True):
    """

    :param evaluator:
    :param batch_size:
    :param num_restarts:
    :param update_interval:
    :param initial_samples: tuple or int. If tuple: (X,Y) defines precollected data to be used for model initialization.
            If int, no precollected data is used and initial_samples defines number of random samples to be generated
            for model initialization.
    :param method: string among "mes", "gpbucb", "mfmes"
    :param fidelities:
    :param use_standard_kernels:
    :return:
    """

    if not method in ("mes", "gpbucb", "mfmes"):
        raise RuntimeError("Wrong method: %s" % method)

    if method == "mfmes" and (fidelities is None or len(fidelities) == 0):
        raise RuntimeError("Fidelities must be given for MF-MES")

    # Set up parameter space
    parameter_space = [
        ContinuousParameter(p["name"], p["bounds"][0], p["bounds"][1])
        for p in evaluator.problem.parameters
    ]

    if method == "mfmes": # Add fidelity parameter if we set up MFMES
        parameter_space.append(InformationSourceParameter(len(fidelities)))

    parameter_space = ParameterSpace(parameter_space)
    number_of_parameters = len(evaluator.problem.parameters)

    # Set up wrapper
    fidelity_evaluator = FidelityEvaluator(evaluator, fidelities)

    # Obtain initial sample
    if type(initial_samples) == int:
        design = RandomDesign(parameter_space)
        initial_x = design.get_samples(initial_samples)
        initial_y = fidelity_evaluator(initial_x)
    else:
        initial_x, initial_y = initial_samples

    # Set up model
    if method != "mfmes":
        model = define_GPmodel_sf(use_standard_kernels, number_of_parameters,
                                  initial_x, initial_y, num_restarts=num_restarts)

    else:
        model = define_GPmodel_mf(number_of_parameters, initial_x, initial_y,
                                  fidelities, num_restarts=num_restarts)

    # Set up BO loop

    if method == "mfmes":
        cost_acquisition = Cost([f["cost"] for f in fidelities])
        #     acquisition          = MultiInformationSourceEntropySearch(model, parameter_space) / cost_acquisition
        acquisition = MultiFidelityMinValueEntropySearch(model, parameter_space) / cost_acquisition
        gradient_optimizer = GradientAcquisitionOptimizer(space=parameter_space)
        optimizer = MultiSourceAcquisitionOptimizer(gradient_optimizer, space=parameter_space)
        point_calculator = GreedyBatchPointCalculator(model=model,
                                                      acquisition=acquisition,
                                                      acquisition_optimizer=optimizer,
                                                      batch_size=batch_size)
    else:
        if method == "mes":
            acquisition = MinValueEntropySearch(model=model, space=parameter_space)

        elif method == "gpbucb":
            acquisition = NegativeLowerConfidenceBound(model=model)

        optimizer = GradientAcquisitionOptimizer(space=parameter_space)
        point_calculator = GreedyBatchPointCalculator(model=model,
                                                      acquisition=acquisition,
                                                      acquisition_optimizer=optimizer,
                                                      batch_size=batch_size)

    bayesopt_loop = BayesianOptimizationLoop(model=model,
                                             space=parameter_space,
                                             acquisition=acquisition,
                                             batch_size=batch_size,
                                             update_interval=update_interval,
                                             candidate_point_calculator=point_calculator)

    stopping_condition = EvaluatorStoppingCondition(evaluator)

    bayesopt_loop.run_loop(fidelity_evaluator, stopping_condition)





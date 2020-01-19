from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from emukit.core.loop.user_function import MultiSourceFunctionWrapper

from emukit.multi_fidelity.models.linear_model import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel import LinearMultiFidelityKernel

from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
from emukit.model_wrappers import GPyModelWrapper

#from emukit.bayesian_optimization.acquisitions.entropy_search import EntropySearch, Cost
from emukit.bayesian_optimization.acquisitions import NegativeLowerConfidenceBound
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.bayesian_optimization.loops.bayesian_optimization_loop import BayesianOptimizationLoop
from emukit.core.loop import FixedIterationsStoppingCondition
from emukit.core.acquisition import Acquisition
from emukit.core.loop.candidate_point_calculators import GreedyBatchPointCalculator
from emukit.core.initial_designs import RandomDesign
from emukit.bayesian_optimization.loops.cost_sensitive_bayesian_optimization_loop import OuterLoop
from emukit.core.loop.model_updaters import FixedIntervalUpdater
from emukit.core.loop.loop_state import create_loop_state, LoopState

from emukit.core.acquisition import IntegratedHyperParameterAcquisition
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.core.loop.stopping_conditions import StoppingCondition
from emukit.bayesian_optimization.acquisitions import EntropySearch, ExpectedImprovement, MaxValueEntropySearch

from emukit.model_wrappers import GPyMultiOutputWrapper

import GPy
from GPy.models import GPRegression

import copy
import numpy as np

class FidelityCollector:
    """
        This class is a callable that obtains simulation results and passes
        them on to emukit. As is convention with emukit, the last column of
        the input data is the fidelity level. If the defined fidelities are
        not None, the case is expected to have this additional column.
    """
    def __init__(self, calibrator, fidelities = None):
        self.calibrator = calibrator
        self.fidelities = fidelities
        self.annotations = { "bo_iteration": 0 }

    def __call__(self, parameters):
        annotations = {}
        annotations.update(self.annotations)

        identifiers = []

        if self.fidelities is None:
            for p in parameters:
                identifiers.append(self.calibrator.schedule(p, annotations = annotations))

        else:
            for p in parameters:
                fidelity = self.fidelities[int(p[-1])]

                annotations.update({ "fidelity_level": fidelity["name"] })
                fidelity_parameters = fidelity["parameters"]

                identifiers.append(self.calibrator.schedule(p[:-1], fidelity_parameters, annotations = annotations))

        self.calibrator.wait(identifiers)
        objectives = [self.calibrator.get(identifier)[0] for identifier in identifiers]

        for identifier in identifiers:
            self.calibrator.cleanup(identifier)

        return np.array(objectives).reshape((parameters.shape[0], -1))

    def update(self, loop, state):
        # @Anastasia: Here one could add additional information that is saved
        # in the output log file. Like things to display the acquisition function
        # etc... This function here is called as a iteration ends callback in
        # the BO loop.

        self.annotations["bo_iterations"] = state.iteration

class CalibratorStoppingCondition(StoppingCondition):
    """
        This is a emukit stopping condition that works exactly as the stopping
        condition for all the other algorithms. I.e., one can define a maximum
        cost or maximum number of iterations when setting up the optimizer.
    """
    def __init__(self, calibrator):
        self.calibrator = calibrator

    def should_stop(self, state):
        return self.calibrator.finished

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
def bo_algorithm(calibrator, batch_size = 4, num_restarts = 1, update_interval = 1, initial_samples = 4, method = "mes", fidelities = None, use_standard_kernels = True):
    if not method in ("mes", "gpbucb", "mfmes"):
        raise RuntimeError("Wrong method: %s" % method)

    if method == "mfmes" and (fidelities is None or len(fidelities) == 0):
        raise RuntimeError("Fidelities must be given for MF-MES")

    # Set up parameter space
    parameter_space = [
        ContinuousParameter(p["name"], p["bounds"][0], p["bounds"][1])
        for p in calibrator.problem.parameters
    ]

    if method == "mfmes": # Add fidelity parameter if we set up MFMES
        parameter_space.append(InformationSourceParameter(len(fidelities)))

    parameter_space = ParameterSpace(parameter_space)
    number_of_parameters = len(calibrator.problem.parameters)

    # Set up wrapper
    collector = FidelityCollector(calibrator, fidelities)

    # Obtain initial samples
    if type(initial_samples) == int:
        design = RandomDesign(parameter_space)
        initial_x = design.get_samples(initial_samples)
        initial_y = collector(initial_x)
    else:
        initial_x, initial_y = initial_samples

    # @ Anstasia: The above lines either generate a set of random samples
    # from scratch (if it is given as an integer, which is the default
    # behaviour right now), but you can also pass a tuple with X and Y
    # to start with them. I implemented this as I did not have the respective
    # file with the initial samples that you use in the notebook. Hence, I was
    # also not sure how much of them you use.

    # Set up model
    if method != "mfmes": # Single fidelity
        kernel = None

        if not use_standard_kernels:
            # @ Anastasia: Here convergence is very poor if I use this kernel from the notebook. Did I do something wrong?
            kernel = GPy.kern.sde_Matern52(number_of_parameters, lengthscale = 1.0, variance = 1.0, ARD = True)
            kernel.variance.constrain_bounded(0.05, 0.1)
            kernel.lengthscale.constrain_bounded(0, 0.1)

        model = GPy.models.GPRegression(initial_x, initial_y, kernel)
        model.Gaussian_noise.variance.fix(1e-3)

    else: # Multi fidelity
        kernel_low = GPy.kern.sde_Matern32(number_of_parameters, lengthscale = 1.0, variance = 0.2, ARD = True)
        kernel_low.variance.constrain_bounded(0.05, 0.1)
        kernel_low.lengthscale.constrain_bounded(0, 0.1)

        kernel_error = GPy.kern.sde_Matern52(number_of_parameters, lengthscale = 1.0, variance = 0.2, ARD = True)
        kernel_error.lengthscale.constrain_bounded(0, 0.1)
        kernel_error.variance.constrain_bounded(10e-5, 5.0 * 10e-5)

        kernel = LinearMultiFidelityKernel([kernel_low, kernel_error])
        kernel.constrain_bounded(0.0, 1.0, 1.0)
        kernel.scale.constrain_bounded(0.0, 0.6)

        model = GPyLinearMultiFidelityModel(initial_x, initial_y, kernel, len(fidelities))
        model.likelihood.Gaussian_noise.fix(5.0 * 1e-3)
        model.likelihood.Gaussian_noise_1.fix(1e-3)

        model = GPyMultiOutputWrapper(
            model, n_outputs = len(fidelities),
            n_optimization_restarts = 1, verbose_optimization = False
        )

    model.optimize()

    model = GPyModelWrapper(model, n_restarts = num_restarts)

    # Set up BO loop
    if method == "mes":
        # @ Anastasia: Not sure here, I cannot find the minentropy_* stuff in the package?
        acquisition = MaxValueEntropySearch(model = model, space = parameter_space)
    elif method == "gpbucb":
        acquisition = NegativeLowerConfidenceBound(model = model)
    elif method == "mfmes":
        # @ Anastasia: Not sure at all here, none of those classes seem to exist anymore?
        pass
        #cost_acquisition = Cost([f["cost"] for f in fidelities])
        #acquisition = MultiFidelityMinValueEntropySearch(model, parameter_space) / cost_acquisition

    optimizer = GradientAcquisitionOptimizer(space = parameter_space)

    if method == "mfmes":
        # @ Anastasia: Could not test since I did not get here with MFMES.
        optimizer = MultiSourceAcquisitionOptimizer(optimizer, space = parameter_space)

    # @ Anastasia: From here I had to get creative. It seems that in the latest
    # pip version of emukit there is no candidate_point_calculator parameter for
    # the BayesianOptimizationLoop. Hence I looked into the code and reconstructed
    # it with a OuterLoop objective, plus I defined the point calculator as you
    # can see below. Is this still all correct now?

    point_calculator = GreedyBatchPointCalculator(
        model = model, acquisition = acquisition,
        acquisition_optimizer = optimizer, batch_size = batch_size)

    model_updaters = FixedIntervalUpdater(model, update_interval)
    loop_state = create_loop_state(model.X, model.Y)

    loop = OuterLoop(point_calculator, model_updaters, loop_state)
    stopping_condition = CalibratorStoppingCondition(calibrator)

    loop.iteration_end_event.append(collector.update)

    loop.run_loop(collector, stopping_condition)

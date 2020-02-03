from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter

from emukit.multi_fidelity.models.linear_model import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel import LinearMultiFidelityKernel

from emukit.model_wrappers import GPyModelWrapper

from emukit.bayesian_optimization.acquisitions import NegativeLowerConfidenceBound
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.bayesian_optimization.loops.bayesian_optimization_loop import BayesianOptimizationLoop
from emukit.core.loop.candidate_point_calculators import GreedyBatchPointCalculator
from emukit.experimental_design.model_free.latin_design import LatinDesign

from emukit.core.loop.stopping_conditions import StoppingCondition
from emukit.bayesian_optimization.acquisitions.minvalue_entropy_search import MinValueEntropySearch, Cost, MultiFidelityMinValueEntropySearch

from emukit.model_wrappers import GPyMultiOutputWrapper
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch
from emukit.core.loop import FixedIterationsStoppingCondition


import GPy
import numpy as np


def bo_algorithm(evaluator, batch_size=4, num_restarts=1, update_interval=1, initial_samples=4, method="mes",
                fidelities=None, use_standard_kernels=False, bo_iterations=50):

    bo = BO(evaluator, batch_size, num_restarts, update_interval, initial_samples, method,
                fidelities=fidelities, use_standard_kernels=use_standard_kernels, bo_iterations=bo_iterations)

    bo.bo_run()


def subdomain_bo_algorithm(evaluator, batch_size=4, num_restarts=1, update_interval=1, initial_samples=4, method="mes",
                fidelities=None, use_standard_kernels=False, subdomain_size=3, num_subdomain_iters=1, bo_iterations=50):

    bo = BO(evaluator, batch_size, num_restarts, update_interval, initial_samples, method,
                fidelities=fidelities, use_standard_kernels=use_standard_kernels, bo_iterations=bo_iterations)
    bo.subdomain_bo_run(subdomain_size, num_subdomain_iters)


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
            Example for MF setting: tests/test_bo -> test_multi_fidelity
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

        self.annotations["bo_iterations"] += 1

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


class BO:
    def __init__(self, evaluator, batch_size=4, num_restarts=1, update_interval=1, initial_samples=4, method="mes",
                 gp_model=None, fidelities=None, use_standard_kernels=True, bo_iterations=50):

        self.evaluator = evaluator
        self.batch_size = batch_size
        self.num_restarts = num_restarts
        self.update_interval = update_interval
        self.method = method
        self.fidelities = fidelities
        self.use_standard_kernels = use_standard_kernels
        self.gp_model = gp_model

        if not self.method in ("mes", "gpbucb", "mfmes"):
            raise RuntimeError("Wrong method: %s" % self.method)

        if self.method == "mfmes" and (self.fidelities is None or len(self.fidelities) == 0):
            raise RuntimeError("Fidelities must be given for MF-MES")

        # Set up parameter space
        self.parameter_space_without_fidelity = [
            ContinuousParameter(p["name"], p["bounds"][0], p["bounds"][1])
            for p in self.evaluator.problem.parameters
        ]

        if self.method == "mfmes":  # Add fidelity parameter if we set up MFMES
            self.parameter_space = ParameterSpace(self.parameter_space_without_fidelity + \
                                    [InformationSourceParameter(len(self.fidelities))])
        else:
            self.parameter_space = ParameterSpace(self.parameter_space_without_fidelity)

        self.num_parameters = len(self.evaluator.problem.parameters)

        # Set up wrapper
        self.fidelity_evaluator = FidelityEvaluator(self.evaluator, self.fidelities)

        # Obtain initial sample
        if type(initial_samples) == int:
            if self.method == "mfmes":
                design = LatinDesign(ParameterSpace(self.parameter_space_without_fidelity))
            else:
                design = LatinDesign(self.parameter_space)
            initial_x = design.get_samples(initial_samples)
            initial_x_low = np.c_[initial_x, np.zeros(len(initial_x))]
            initial_x_high = np.c_[initial_x, np.ones(len(initial_x))]
            self.initial_x = np.vstack((initial_x_low, initial_x_high))
            self.initial_y = self.fidelity_evaluator(self.initial_x)
        else:
            self.initial_x, self.initial_y = initial_samples

        # Set up model
        if self.gp_model is None:
            if method != "mfmes":
                self.gp_model = self.define_gpmodel_sf(self.num_parameters,
                                                       self.initial_x, self.initial_y, num_restarts=self.num_restarts)
            else:
                self.gp_model = self.define_gpmodel_mf(self.num_parameters, self.initial_x, self.initial_y,
                                                       fidelities, num_restarts=num_restarts)

        self.stopping_condition = FixedIterationsStoppingCondition(i_max=bo_iterations)

    @staticmethod
    def define_gpmodel_mf(num_parameters, initial_x, initial_y, fidelities, num_restarts=1):

        kernel_low = GPy.kern.sde_Matern32(num_parameters, lengthscale=1.0, variance=0.2, ARD=True)
        kernel_low.variance.constrain_bounded(0.05, 0.1)
        kernel_low.lengthscale.constrain_bounded(0, 0.1)

        kernel_error = GPy.kern.sde_Matern52(num_parameters, lengthscale=1.0, variance=0.2, ARD=True)
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

    def define_gpmodel_sf(self, num_parameters, initial_x, initial_y, num_restarts=1):
        kernel = None
        if not self.use_standard_kernels:
            # @ Anastasia: Here convergence is very poor if I use this kernel from the notebook. Did I do something wrong?
            kernel = GPy.kern.sde_Matern52(num_parameters, lengthscale=1.0, variance=1.0, ARD=True)
            kernel.variance.constrain_bounded(0.05, 0.1)
            kernel.lengthscale.constrain_bounded(0, 0.1)

        model = GPy.models.GPRegression(initial_x, initial_y, kernel)
        model.Gaussian_noise.variance.fix(1e-3)

        model = GPyModelWrapper(model, n_restarts=num_restarts)
        model.optimize()

        return model

    def get_bo_loop(self, model, parameter_space):

        if self.method == "mfmes":
            cost_acquisition = Cost([f["cost"] for f in self.fidelities])
            # acquisition  = MultiInformationSourceEntropySearch(model, parameter_space) / cost_acquisition
            acquisition = MultiFidelityMinValueEntropySearch(model, parameter_space) / cost_acquisition
            gradient_optimizer = GradientAcquisitionOptimizer(space=parameter_space)
            optimizer = MultiSourceAcquisitionOptimizer(gradient_optimizer, space=parameter_space)
            point_calculator = GreedyBatchPointCalculator(model=model,
                                                          acquisition=acquisition,
                                                          acquisition_optimizer=optimizer,
                                                          batch_size=self.batch_size)
        else:
            if self.method == "mes":
                acquisition = MinValueEntropySearch(model=model, space=parameter_space)

            elif self.method == "gpbucb":
                acquisition = NegativeLowerConfidenceBound(model=model)

            optimizer = GradientAcquisitionOptimizer(space=parameter_space)
            point_calculator = GreedyBatchPointCalculator(model=model,
                                                          acquisition=acquisition,
                                                          acquisition_optimizer=optimizer,
                                                          batch_size=self.batch_size)

        bayesopt_loop = BayesianOptimizationLoop(model=model,
                                                 space=parameter_space,
                                                 acquisition=acquisition,
                                                 batch_size=self.batch_size,
                                                 update_interval=self.update_interval,
                                                 candidate_point_calculator=point_calculator)

        return bayesopt_loop

    def bo_run(self, model=None):

        # Set up BO loop
        if model is None:
            model = self.gp_model
        bo_loop = self.get_bo_loop(model, self.parameter_space)
        bo_loop.run_loop(self.fidelity_evaluator, self.stopping_condition)

    def subdomain_bo_run(self, subdomain_size=1, num_subdomain_iters=20):
        """

        :param subdomain_size: int, size of subdomain to perform iterative BO on
        :param num_subdomain_iters: int,
        :return:
        """

        values_ = ParameterSpace(self.parameter_space_without_fidelity).sample_uniform(point_count=1)[0]
        for k in range(num_subdomain_iters):

            parameter_space = self.parameter_space
            x, y = self.initial_x, self.initial_y
            coordinates = np.arange(self.num_parameters)
            j = 0
            while len(coordinates) > 0:
                subdomain = np.random.choice(coordinates, min(len(coordinates), subdomain_size))
                parameter_space = self.update_param_space(parameter_space, subdomain, values_)

                if self.method != "mfmes":
                    gp_model = self.define_gpmodel_sf(self.num_parameters, x, y, num_restarts=self.num_restarts)
                else:
                    gp_model = self.define_gpmodel_mf(self.num_parameters, x, y,
                                                     self.fidelities, num_restarts=self.num_restarts)

                bo_loop = self.get_bo_loop(gp_model, parameter_space)
                bo_loop.run_loop(self.fidelity_evaluator, self.stopping_condition)

                # update data for the next round
                x, y = bo_loop.loop_state.X,  bo_loop.loop_state.Y

                values_[subdomain] = bo_loop.get_results().minimum_location[subdomain]
                coordinates = np.delete(coordinates, subdomain)
                j += 1

    def update_param_space(self, parameter_space, subdomain, values):
        """

        :param parameter_space:
        :param subdomain: optimization subdomain
        :param values: best values for the whole domain
        :return:
        """

        calibrated_params = []
        fidelity_param = []

        for p in parameter_space.parameters:
            if p.name == 'source':
                fidelity_param.append(p)
            else:
                calibrated_params.append(p)

        new_parameter_space = [
                                ContinuousParameter(p.name, p.bounds[0][0], p.bounds[0][1]) if i in subdomain
                                else ContinuousParameter(p.name, values[i], values[i])
                                for i, p in enumerate(calibrated_params)
                                ]
        return ParameterSpace(new_parameter_space + fidelity_param)














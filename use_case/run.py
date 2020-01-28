import problems, matsim, octras.simulation, octras.optimization
import os, subprocess
import numpy as np


def verify_configuration(configuration):
    available_sample_sizes = ("1pm", "1pct", "10pct", "25pct")
    available_decision_variables = ("constants", "vots", "all")
    available_problems = ("total_mode_share", "car_travel_time", "mode_share_by_distance")
    available_objectives = ("l2", "hellinger")

    # Get sub-configuration objects
    simulation_configuration = configuration["simulation"] if "simulation" in configuration else {}
    configuration["simulation"] = simulation_configuration

    java_configuration = configuration["java"] if "java" in configuration else {}
    configuration["java"] = java_configuration

    calibration_configuration = configuration["calibration"] if "calibration" in configuration else {}
    configuration["calibration"] = calibration_configuration

    scheduler_configuration = configuration["scheduler"] if "scheduler" in configuration else {}
    configuration["scheduler"] = scheduler_configuration

    optimization_configuration = configuration["optimization"] if "optimization" in configuration else {}
    configuration["optimization"] = optimization_configuration

    # Verify working directory
    if not "working_directory" in configuration:
        raise RuntimeError("Working directory must be set")

    working_directory = configuration["working_directory"]

    if not os.path.isdir(working_directory):
        raise RuntimeError("Working directory is not valid")

    # Verify simulation path
    if not "simulation_path" in configuration:
        raise RuntimeError("Simulation path must be set")

    simulation_path = configuration["simulation_path"]

    if not os.path.exists("%s/switzerland-1.0.5.jar" % simulation_path):
        raise RuntimeError("Simulation path does not seem to be valid.")

    # Verify Java
    java_path = java_configuration["path"] if "path" in java_configuration else "java"
    java_configuration["path"] = java_path

    try:
        value = subprocess.check_call([java_path, "-version"])
    except subprocess.CalledProcessError as e:
        value = 1

    if not value == 0:
        raise RuntimeError("Java seems to be invalid")

    # Verify sample size
    if not "sample_size" in simulation_configuration or not simulation_configuration["sample_size"] in available_sample_sizes:
        raise RuntimeError("Sample size size must be in (%s)" % ", ".join(available_sample_sizes))

    sample_size = simulation_configuration["sample_size"]

    # Verify reference sample size
    reference_sample_size = "25pct"

    if "reference_sample_size" in calibration_configuration:
        reference_sample_size = calibration_configuration["reference_sample_size"]

    if not reference_sample_size in available_sample_sizes:
        raise RuntimeError("Reference sample size must be in (%s)" % ", ".join(available_sample_sizes))

    calibration_configuration["reference_sample_size"] = reference_sample_size
    calibration_configuration["reference_path"] = "%s/zurich_%s/reference.csv" % (simulation_path, reference_sample_size)

    if not os.path.exists(calibration_configuration["reference_path"]):
        raise RuntimeError("Reference path does not exist: %s" % calibration_configuration["reference_path"])

    # Verify decision variables
    if not "decision_variables" in calibration_configuration or not calibration_configuration["decision_variables"] in available_decision_variables:
        raise RuntimeError("Decision variables must be in (%s)" % ", ".join(available_decision_variables))

    # Verify objective function
    if not "objective" in calibration_configuration or not calibration_configuration["objective"] in available_objectives:
        raise RuntimeError("Objective must be in (%s)" % ", ".join(available_objectives))

    # Verify states for the problem
    if not "problem" in calibration_configuration or not calibration_configuration["problem"] in available_problems:
        raise RuntimeError("Problem must be in (%s)" % ", ".join(available_problems))

    # Verify algorithm
    if not "algorithm" in optimization_configuration:
        raise RuntimeError("Algorithm must be in (%s)" % ", ".join('available_algorithms'))

    algorithm = optimization_configuration["algorithm"]

def setup_problem(configuration):
    configuration = configuration["calibration"]

    # Set up decision variables
    if configuration["decision_variables"] == "constants":
        parameters = [ # Only the mode-specific constants will be varied
            { "name": "car.alpha_u", "initial": -0.1, "bounds": (-2.0, 2.0) },
            { "name": "bike.alpha_u", "initial": -0.1, "bounds": (-2.0, 2.0) },
            { "name": "walk.alpha_u", "initial": -0.1, "bounds": (-2.0, 2.0) }
        ]
    elif configuration["decision_variables"] == "vots":
        parameters = [ # Only the mode-specific VOTs will be varied
            { "name": "car.betaTravelTime_u_min", "initial": -0.1, "bounds": (-1.0, 0.0) },
            { "name": "pt.betaInVehicleTime_u_min", "initial": -0.1, "bounds": (-1.0, 0.0) },
            { "name": "pt.betaWaitingTime_u_min", "initial": -0.1, "bounds": (-1.0, 0.0) },
            { "name": "pt.betaAccessEgressTime_u_min", "initial": -0.1, "bounds": (-1.0, 0.0) },
            { "name": "bike.betaTravelTime_u_min", "initial": -0.1, "bounds": (-1.0, 0.0) },
            { "name": "walk.betaTravelTime_u_min", "initial": -0.1, "bounds": (-1.0, 0.0) },
        ]
    elif configuration["decision_variables"] == "all":
        parameters = [ # All available choice parameters will be varied
            { "name": "car.alpha_u", "initial": -0.1, "bounds": (-2.0, 2.0) },
            { "name": "car.betaTravelTime_u_min", "initial": -0.1, "bounds": (-1.0, 0.0) },

            { "name": "pt.betaInVehicleTime_u_min", "initial": -0.1, "bounds": (-1.0, 0.0) },
            { "name": "pt.betaWaitingTime_u_min", "initial": -0.1, "bounds": (-1.0, 0.0) },
            { "name": "pt.betaAccessEgressTime_u_min", "initial": -0.1, "bounds": (-1.0, 0.0) },
            { "name": "pt.betaLineSwitch_u", "initial": -0.1, "bounds": (-1.0, 0.0) },

            { "name": "bike.alpha_u", "initial": -0.1, "bounds": (-2.0, 2.0) },
            { "name": "bike.betaTravelTime_u_min", "initial": -0.1, "bounds": (-1.0, 0.0) },
            { "name": "bike.betaAgeOver18_u_a", "initial": -0.1, "bounds": (-1.0, 0.0) },

            { "name": "walk.alpha_u", "initial": -0.1, "bounds": (-2.0, 2.0) },
            { "name": "walk.betaTravelTime_u_min", "initial": -0.1, "bounds": (-1.0, 0.0) },
        ]

    # Set up objective
    if configuration["objective"] == "l2":
        objective_calculator = problems.l2_distance
    elif configuration["objective"] == "hellinger":
        objective_calculator = problems.hellinger_distance

    # Set up problem state
    if configuration["problem"] == "total_mode_share":
        state_names = modes = ["car", "pt", "bike", "walk"]
        state_calculator = problems.TotalModeShare(modes)
    elif configuration["problem"] == "car_travel_time":
        bounds = np.array((322, 602, 1029, 1961))
        state_names = ["%dmin" % bound for bound in bounds]
        state_calculator = problems.TravelTimeDistribution(bounds, "car")
    elif configuration["problem"] == "mode_share_by_distance":
        mode_bounds = [ # TODO
            { "mode": "car", "bounds": [2061.9277893732583, 4441.935709111793, 8153.668874618363, 15494.8883624554] },
            { "mode": "pt", "bounds": [3279.9146782441344, 6054.286007863699, 10627.070199423772, 20268.556604472567] },
            { "mode": "bike", "bounds": [642.4853052235655, 1172.1277916677855, 2014.6189714186648, 3617.8746794815843] },
            { "mode": "walk", "bounds": [237.9035518860532, 388.7086955549104, 585.9770729303324, 926.1667506448283] },
        ]
        state_calculator = problems.ModeShareByDistance(mode_bounds)

        state_names = []

        for item in mode_bounds:
            state_names += ["%s_%d" % (item["mode"], b) for b in item["bounds"]]

    # Verify problem instance
    return problems.TripBasedProblem(
        problem_name = configuration["problem"],

        state_calculator = state_calculator,
        state_names = state_names,

        objective_calculator = objective_calculator,
        parameters = parameters,

        reference_path = configuration["reference_path"]
    ), parameters

def setup_simulator(configuration):
    simulator_configuration = {
        "java_path": configuration["java"]["path"],
        "working_directory": configuration["working_directory"],
        "simulation_path": configuration["simulation_path"],
    }

    if "memory" in configuration["simulation"]:
        simulator_configuration["java_memory"] = configuration["simulation"]["memory"]

    if "number_of_threads" in configuration["simulation"]:
        simulator_configuration["number_of_threads"] = configuration["simulation"]["number_of_threads"]

    if not "iterations" in configuration["simulation"]:
        raise RuntimeError("Number of iterations must be set!")

    simulator_configuration["iterations"] = configuration["simulation"]["iterations"]
    simulator_configuration["sample_size"] = configuration["simulation"]["sample_size"]

    return matsim.MATSimSimulator(simulator_configuration)

def setup_scheduler(simulator, configuration):
    arguments = {}

    if "ping_time" in configuration["scheduler"]:
        arguments["ping_time"] = configuration["scheduler"]["ping_time"]

    if "number_of_runners" in configuration["scheduler"]:
        arguments["number_of_runners"] = configuration["scheduler"]["number_of_runners"]

    return octras.simulation.Scheduler(simulator, **arguments)

def setup_optimizer(scheduler, problem, configuration):
    arguments = {}

    if "output_path" in configuration["calibration"]:
        arguments["log_path"] = configuration["calibration"]["output_path"]

    if "maximum_evaluations" in configuration["calibration"]:
        arguments["maximum_evaluations"] = configuration["calibration"]["maximum_evaluations"]

    if "maximum_cost" in configuration["calibration"]:
        arguments["maximum_cost"] = configuration["calibration"]["maximum_cost"]

    return octras.optimization.Optimizer(
        scheduler, problem, **arguments
    )

def run_experiment(optimizer, parameters, configuration):
    algorithm = configuration["optimization"]["algorithm"]

    if algorithm == "random_walk":
        return run_random_walk(optimizer, parameters)
    elif algorithm == "fdsa" or algorithm == "spsa":
        return run_fdsa_spsa(algorithm, optimizer, configuration["optimization"])
    elif algorithm == "cma_es":
        return run_cma_es(optimizer, configuration["optimization"])
    elif algorithm == "opdyts":
        return run_opdyts(optimizer, configuration["optimization"])
    elif algorithm == "bo":
        return run_bo(optimizer, configuration["optimization"])
    else:
        raise RuntimeError("Unknown algorithm: %s" % algorithm)

def run_random_walk(optimizer, parameters):
    bounds = [
        parameter["bounds"] if "bounds" in parameter else (None, None)
        for parameter in parameters
    ]

    from octras.algorithms.random_walk import random_walk_algorithm
    random_walk_algorithm(optimizer, bounds)

def run_fdsa_spsa(algorithm, optimizer, configuration):
    arguments = {}

    if not "perturbation_factor" in configuration:
        raise RuntimeError("Perturbation factor must be given for FDSA")

    if not "gradient_factor" in configuration:
        raise RuntimeError("Gradient factor must be given for FDSA")

    arguments["perturbation_factor"] = configuration["perturbation_factor"]
    arguments["gradient_factor"] = configuration["gradient_factor"]

    if "perturbation_exponent" in configuration:
        arguments["perturbation_exponent"] = configuration["perturbation_exponent"]

    if "gradient_exponent" in configuration:
        arguments["gradient_exponent"] = configuration["gradient_exponent"]

    if "gradient_constant" in configuration:
        arguments["gradient_constant"] = configuration["gradient_constant"]

    if "compute_objective" in configuration:
        arguments["compute_objective"] = configuration["compute_objective"]

    if algorithm == "fdsa":
        from octras.algorithms.fdsa import fdsa_algorithm
        fdsa_algorithm(optimizer, **arguments)
    elif algorithm == "spsa":
        from octras.algorithms.spsa import spsa_algorithm
        spsa_algorithm(optimizer, **arguments)

def run_cma_es(optimizer, configuration):
    arguments = {}

    if "candidate_set_size" in configuration:
        arguments["candidate_set_size"] = configuration["candidate_set_size"]

    if "initial_step_size" in configuration:
        arguments["initial_step_size"] = configuration["initial_step_size"]

    from octras.algorithms.cma_es import cma_es_algorithm
    cma_es_algorithm(optimizer, **arguments)

def run_opdyts(optimizer, configuration):
    arguments = {}

    if not "candidate_set_size" in configuration:
        raise RuntimeError("Candidate set size must be set for opdyts")

    if not "transition_iterations" in configuration:
        raise RuntimeError("Transition iterations be set for opdyts")

    if not "number_of_transitions" in configuration:
        raise RuntimeError("Number of transitions must be set for opdyts")

    if not "adaptation_weight" in configuration:
        raise RuntimeError("Perturbation length must be set for opdyts")

    if not "perturbation_length" in configuration:
        raise RuntimeError("Adaptation weight must be set for opdyts")

    arguments["candidate_set_size"] = configuration["candidate_set_size"]
    arguments["transition_iterations"] = configuration["transition_iterations"]
    arguments["number_of_transitions"] = configuration["number_of_transitions"]
    arguments["adaptation_weight"] = configuration["adaptation_weight"]
    arguments["perturbation_length"] = configuration["perturbation_length"]

    from octras.algorithms.opdyts import opdyts_algorithm
    opdyts_algorithm(optimizer, **arguments)

def run_bo(optimizer, configuration):
    arguments = {}

    if not "method" in configuration:
        raise RuntimeError("Method must be set for Bayesian optimization")

    arguments["method"] = configuration["method"]

    if "batch_size" in configuration:
        arguments["batch_size"] = configuration["batch_size"]

    if "initial_samples" in configuration:
        arguments["initial_samples"] = configuration["initial_samples"]

    if arguments["method"] == "mfmes":
        if not "fidelities" in configuration or not configuration["fidelities"] in ("sample_size", "iterations"):
            raise RuntimeError("Fidelity must be set for MF-MES. Select from (sample_size, iterations).")

        if configuration["fidelities"] == "sample_size":
            fidelities = [
                { "name": "1pm", "cost": 0.001, "parameters": { "sample_size": "1pm" } },
                { "name": "1pct", "cost": 0.01, "parameters": { "sample_size": "1pct" } },
                { "name": "10pct", "cost": 0.1, "parameters": { "sample_size": "10pct" } }
            ]

        if configuration["fidelities"] == "iterations":
            fidelities = [
                { "name": "10it", "cost": 1, "parameters": { "iterations": 10 } },
                # { "name": "40it", "cost": 40, "parameters": { "iterations": 40 } },
                { "name": "40it", "cost": 4, "parameters": { "iterations": 40} },
            ]

        arguments["fidelities"] = fidelities

    from octras.algorithms.bo import bo_algorithm, subdomain_bo_algorithm
    if "subdomain_bo" in configuration and configuration["subdomain_bo"]:
        arguments["subdomain_size"] = 3
        arguments["num_subdomain_iters"] = 10
        subdomain_bo_algorithm(optimizer, **arguments)
    else:
        bo_algorithm(optimizer, **arguments)

def parse_arguments(args, configuration):
    if len(args) % 2 != 0:
        raise RuntimeError("Wrong number of arguments")

    for index in range(0, len(args), 2):
        name, value = args[index], args[index + 1]

        if not name.startswith("--"):
            raise RuntimeError("Wrong format: %s" % name)

        name = name[2:]
        value = int(value)

        partial_configuration = configuration
        parts = name.split(".")

        for part in parts[:-1]:
            if not part in partial_configuration:
                partial_configuration[part] = {}

            partial_configuration = partial_configuration[part]

        partial_configuration[parts[-1]] = value

if __name__ == "__main__":
    import sys, yaml

    import logging
    logging.basicConfig(level = logging.INFO)

    if len(sys.argv) < 2:
        raise RuntimeError("Expecting path to config file as first argument")

    with open(sys.argv[1]) as f:
        configuration = yaml.load(f, Loader = yaml.SafeLoader)

    parse_arguments(sys.argv[2:], configuration)
    verify_configuration(configuration)

    problem, parameters = setup_problem(configuration)
    simulator = setup_simulator(configuration)
    scheduler = setup_scheduler(simulator, configuration)
    optimizer = setup_optimizer(scheduler, problem, configuration)

    run_experiment(optimizer, parameters, configuration)



    #problem = setup_problem(configuration)

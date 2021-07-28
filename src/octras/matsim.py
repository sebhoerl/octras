from octras import Simulator

import os, shutil
import subprocess as sp
import pandas as pd
import numpy as np
import glob
import json

import logging
import deep_merge

logger = logging.getLogger("octras")

class ConvergenceHandler:
    def may_terminate(self, path, iteration):
        pass

    def do_terminate(self, path, iteration):
        pass

class MATSimSimulator(Simulator):
    """
        Defines a wrapper around a standard MATSim simulation
    """

    def __init__(self, working_directory, parameters, convergence_handler = None):
        if not os.path.exists(working_directory):
            raise RuntimeError("Working directory does not exist: %s" % working_directory)

        self.working_directory = os.path.realpath(working_directory)
        self.parameters = parameters
        self.convergence_handler = convergence_handler

        if not "memory" in self.parameters:
            self.parameters["memory"] = "10G"

        if not "java" in self.parameters:
            self.parameters["java"] = "java"

        if not "arguments" in self.parameters:
            self.parameters["arguments"] = []

        if not "config" in self.parameters:
            self.parameters["config"] = {}

        self.simulations = {}

    def run(self, identifier, parameters):
        if identifier in self.simulations:
            raise RuntimeError("A simulation with identifier %s already exists." % identifier)

        # Prepare the working space
        simulation_path = "%s/%s" % (self.working_directory, identifier)

        if os.path.exists(simulation_path):
            shutil.rmtree(simulation_path)

        os.mkdir(simulation_path)

        # Merge config
        config = dict()
        config.update(self.parameters["config"])

        if "config" in parameters:
            config.update(parameters["config"])

        # Merge arguments
        arguments = []
        arguments += self.parameters["arguments"]

        if "arguments" in parameters:
            arguments += parameters["arguments"]

        # A specific random seed is requested
        if "random_seed" in parameters or "random_seed" in self.parameters:
            random_seed = None
            if "random_seed" in self.parameters: random_seed = self.parameters["random_seed"]
            if "random_seed" in parameters: random_seed = parameters["random_seed"]

            if "global.random_seed" in config:
                logger.warn("Overwriting 'global.random_seed' for simulation %s" % identifier)

            config["global.random_seed"] = random_seed

        # It is requested to restart from a certain existing simulation output
        if "restart" in parameters:
            if "plans.inputPlansFile" in config:
                logger.warn("Overwriting 'plans.inputPlansFile' for simulation %s" % identifier)

            if "controler.firstIteration" in config:
                logger.warn("Overwriting 'controler.firstIteration' for simulation %s" % identifier)

            restart_path = "%s/%s" % (self.working_directory, parameters["restart"])

            # Find the iteration at which to start
            df_stopwatch = pd.read_csv("%s/output/stopwatch.txt", sep = ";")
            first_iteration = df_stopwatch["Iteration"].max()

            config["plans.inputPlansFile"] = "%s/output/output_plans.xml.gz" % restart_path
            config["controler.firstIteration"] = first_iteration

        # A certain number of iterations is requested
        iterations = None

        if "iterations" in parameters or "iterations" in self.parameters:
            if "iterations" in self.parameters: iterations = self.parameters["iterations"]
            if "iterations" in parameters: iterations = parameters["iterations"]

            if "controler.lastIteration" in config:
                logger.warn("Overwriting 'controler.lastIteration' for simulation %s" % identifier)

            if "controler.writeEventsInterval" in config:
                logger.warn("Overwriting 'controler.writeEventsInterval' for simulation %s" % identifier)

            if "controler.writePlansInterval" in config:
                logger.warn("Overwriting 'controler.writePlansInterval' for simulation %s" % identifier)

            last_iteration = iterations

            # In case firstIteration is set, we need to add the number here
            if "controler.firstIteration" in config:
                last_iteration += parameters["controler.firstIteration"]

            config["controler.lastIteration"] = last_iteration
            config["controler.writeEventsInterval"] = last_iteration
            config["controler.writePlansInterval"] = last_iteration

        # Output directory is standardized so we know where the files are
        if "controler.outputDirectory" in config:
            logger.warn("Overwriting 'controler.outputDirectory' for simulation %s" % identifier)

        config["controler.outputDirectory"] = "%s/output" % simulation_path

        # Construct command line arguments
        if not "class_path" in parameters and not "class_path" in self.parameters:
            raise RuntimeError("Parameter 'class_path' must be set for the MATSim simulator.")

        if not "main_class" in parameters and not "main_class" in self.parameters:
            raise RuntimeError("Parameter 'main_class' must be set for the MATSim simulator.")

        class_path = None
        if "class_path" in self.parameters: class_path = self.parameters["class_path"]
        if "class_path" in parameters: class_path = parameters["class_path"]

        main_class = None
        if "main_class" in self.parameters: main_class = self.parameters["main_class"]
        if "main_class" in parameters: main_class = parameters["main_class"]

        java = self.parameters["java"]
        if "java" in parameters: java = parameters["java"]

        memory = self.parameters["memory"]
        if "memory" in parameters: memory = parameters["memory"]

        arguments = [
            java, "-Xmx%s" % memory,
            "-cp", class_path, main_class
        ] + arguments

        for key, value in config.items():
            arguments += ["--config:%s" % key, str(value)]

        arguments = [str(a) for a in arguments]

        stdout = open("%s/simulation_output.log" % simulation_path, "w+")
        stderr = open("%s/simulation_error.log" % simulation_path, "w+")

        logger.info("Starting simulation %s:" % identifier)
        logger.info(" ".join(arguments))

        self.simulations[identifier] = {
            "process": sp.Popen(arguments, stdout = stdout, stderr = stderr),
            "arguments": arguments, "status": "running", "progress": -1,
            "iterations": iterations,
            "convergence_sequence_may": -1,
            "convergence_sequence_do": -1
        }

    def _handle_convergence(self, identifier):
        simulation_path = "%s/%s/output" % (self.working_directory, identifier)
        terminate = False

        if os.path.exists("%s/tmp/convergence_output.json" % simulation_path):
            with open("%s/tmp/convergence_output.json" % simulation_path) as f:
                if self.convergence_handler is None:
                    raise RuntimeError("MATSim asks for convergence but no handler is defined")

                try:
                    output = json.load(f)
                except json.JSONDecodeError:
                    # Most likely file write synchronisation issue
                    return False

                iteration = self._get_iteration(identifier)
                response = None

                if output["signal"] == "mayTerminate":
                    if self.simulations[identifier]["convergence_sequence_may"] < output["sequence"]:
                        response = bool(self.convergence_handler.may_terminate(simulation_path, iteration + 1))
                        self.simulations[identifier]["convergence_sequence_may"] = output["sequence"]
                elif output["signal"] == "doTerminate":
                    if self.simulations[identifier]["convergence_sequence_do"] < output["sequence"]:
                        response = bool(self.convergence_handler.do_terminate(simulation_path, iteration))
                        self.simulations[identifier]["convergence_sequence_do"] = output["sequence"]
                        terminate = response
                else:
                    raise RuntimeError("Unknown signal")

            if not response is None:
                with open("%s/tmp/convergence_input.json" % simulation_path, "w+") as f:
                    json.dump(dict(
                        sequence = output["sequence"],
                        response = response
                    ), f)

        return terminate

    def _ping(self):
        for identifier, simulation in self.simulations.items():
            if simulation["status"] == "running":
                return_code = simulation["process"].poll()

                if return_code is None:
                    # Still running!
                    iteration = self._get_iteration(identifier)

                    if self._handle_convergence(identifier):
                        logger.info("Convergence criterion says simulation {} should terminate.".format(
                            identifier
                        ))

                    if iteration > simulation["progress"]:
                        simulation["progress"] = iteration

                        logger.info("Running simulation {} ... ({}/{} iterations)".format(
                            identifier, iteration, "?" if simulation["iterations"] is None else simulation["iterations"]
                        ))

                elif return_code == 0:
                    # Finished
                    logger.info("Finished simulation {}".format(identifier))
                    simulation["status"] = "done"
                else:
                    # Errorerd
                    del self.simulations[identifier]
                    raise RuntimeError("Error running simulation {}. See {}/{}/simulation_error.log".format(identifier, self.working_directory, identifier))

    def _get_iteration(self, identifier):
        stopwatch_paths = glob.glob("%s/%s/output/*stopwatch.txt" % (self.working_directory, identifier))

        if len(stopwatch_paths) > 0 and os.path.isfile(stopwatch_paths[0]):
            try:
                df = pd.read_csv(stopwatch_paths[0], sep = "\t")

                if len(df) > 0:
                    return df["Iteration"].max()
            except:
                pass

        return -1

    def ready(self, identifier):
        self._ping()
        return self.simulations[identifier]["status"] == "done"

    def get(self, identifier):
        if not self.ready(identifier):
            raise RuntimeError("Simulation %s is not ready to obtain result." % identifier)

        simulation_path = "%s/%s" % (self.working_directory, identifier)
        return "%s/output" % simulation_path

    def clean(self, identifier):
        simulation_path = "%s/%s" % (self.working_directory, identifier)
        shutil.rmtree(simulation_path)

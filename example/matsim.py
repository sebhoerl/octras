from simulator import Simulator

import os, shutil
import subprocess as sp
import pandas as pd
import numpy as np
import glob

import logging
import deep_merge

logger = logging.getLogger(__name__)

class MATSimSimulator(Simulator):
    """
        Defines a wrapper around a standard MATSim simulation.
    """

    def __init__(self, working_directory, **parameters):
        self.working_directory = working_directory
        self.parameters = parameters

        if not "memory" in self.parameters:
            self.parameters["memory"] = "10G"

        if not "java" in self.parameters:
            self.parameters["java"] = "java"

        if not "arguments" in self.parameters:
            self.parameters["arguments"] = []

        if not "prefix_arguments" in self.parameters:
            self.parameters["prefix_arguments"] = []

        if not "postfix_arguments" in self.parameters:
            self.parameters["postfix_arguments"] = []

        if not "config" in self.parameters:
            self.parameters["config"] = {}

        self.simulations = {}

    def run(self, identifier, parameters):
        """
            Runs a MATSim simulation.
        """
        if identifier in self.simulations:
            raise RuntimeError("A simulation with identifier %s already exists." % identifier)

        simulation_path = "%s/%s" % (self.working_directory, identifier)

        if os.path.exists(simulation_path):
            shutil.rmtree(simulation_path)

        os.mkdir(simulation_path)

        simulation_parameters = {}
        simulation_parameters = deep_merge.merge(simulation_parameters, self.parameters)
        simulation_parameters = deep_merge.merge(simulation_parameters, parameters)
        parameters = simulation_parameters

        # Rewrite configuration
        if "iterations" in parameters:
            if "controler.lastIteration" in parameters["config"]:
                logger.warn("Overwriting 'controler.lastIteration' for simulation %s" % identifier)

            if "controler.writeEventsInterval" in parameters["config"]:
                logger.warn("Overwriting 'controler.writeEventsInterval' for simulation %s" % identifier)

            if "controler.writePlansInterval" in parameters["config"]:
                logger.warn("Overwriting 'controler.writePlansInterval' for simulation %s" % identifier)

            parameters["config"]["controler.lastIteration"] = parameters["iterations"]
            parameters["config"]["controler.writeEventsInterval"] = parameters["iterations"]
            parameters["config"]["controler.writePlansInterval"] = parameters["iterations"]

        if "random_seed" in parameters:
            if "global.random_seed" in parameters["config"]:
                logger.warn("Overwriting 'global.random_seed' for simulation %s" % identifier)

            parameters["config"]["global.random_seed"] = parameters["random_seed"]

        if "restart" in parameters:
            if "plans.inputPlansFile" in parameters["config"]:
                logger.warn("Overwriting 'plans.inputPlansFile' for simulation %s" % identifier)

            restart_path = "%s/%s" % (self.working_directory, parameters["restart"])
            parameters["config"]["plans.inputPlansFile"] = "%s/output/output_plans.xml.gz" % restart_path

        if "controler.outputDirectory" in parameters["config"]:
            logger.warn("Overwriting 'controler.outputDirectory' for simulation %s" % identifier)

        parameters["config"]["controler.outputDirectory"] = "%s/output" % simulation_path

        # Construct command line arguments
        if not "class_path" in parameters:
            raise RuntimeError("Parameter 'class_path' must be set for the MATSim simulator.")

        if not "main_class" in parameters:
            raise RuntimeError("Parameter 'main_class' must be set for the MATSim simulator.")

        arguments = [
            parameters["java"], "-Xmx%s" % parameters["memory"],
            "-cp", parameters["class_path"], parameters["main_class"]
        ] + parameters["prefix_arguments"] + parameters["arguments"]

        for key, value in parameters["config"].items():
            arguments += ["--config:%s" % key, str(value)]

        arguments += parameters["postfix_arguments"]

        stdout = open("%s/simulation_output.log" % simulation_path, "w+")
        stderr = open("%s/simulation_error.log" % simulation_path, "w+")

        logger.info("Starting simulation %s:" % identifier)
        logger.info(" ".join(arguments))

        self.simulations[identifier] = {
            "process": sp.Popen(arguments, stdout = stdout, stderr = stderr),
            "arguments": arguments, "status": "running", "progress": -1,
            "iterations": parameters["iterations"] if "iterations" in parameters else None
        }

    def _ping(self):
        for identifier, simulation in self.simulations.items():
            if simulation["status"] == "running":
                return_code = simulation["process"].poll()

                if return_code is None:
                    # Still running!
                    iteration = self._get_iteration(identifier)

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

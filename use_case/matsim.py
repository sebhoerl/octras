import octras.simulation
import octras.optimization

import os, shutil
import subprocess as sp
import pandas as pd
import numpy as np

import logging

logger = logging.getLogger(__name__)

class Status:
    INITIAL = 0
    SIMULATION = 1
    ANALSYIS = 2
    DONE = 3

class MATSimSimulator(octras.simulation.Simulator):
    def __init__(self, parameters):
        self.parameters = {
            "java_path": "java",
            "java_memory": "10G",
            "number_of_threads": 4
        }
        self.parameters.update(parameters)

        if not "working_directory" in self.parameters:
            raise RuntimeError("Parameter working_directory must be set")

        self.working_directory = os.path.realpath(self.parameters["working_directory"])

        if not "simulation_path" in self.parameters:
            raise RuntimeError("Parameter simulation_path must be set")

        if not os.path.isdir(self.working_directory):
            raise RuntimeError("Working directory must exist: %s" % self.working_directory)

        self.parameters.update({
            "class_path": "%s/astra_2018_002-1.0.0.jar" % self.parameters["simulation_path"],
            "config_path": "%s/zurich_{sample_size}/zurich_config.xml" % self.parameters["simulation_path"]
        })

        self.identifier_mapping = {}

        self.simulation_parameters = {}
        self.status = {}
        self.process = {}
        self.progress = {}

    def _create_internal_identifier(self, identifier):
        internal_identifier = identifier
        path = "%s/%s" % (self.working_directory, internal_identifier)

        index = 1

        while os.path.exists(path):
            internal_identifier = "%s_%d" % (identifier, index)
            path = "%s/%s" % (self.working_directory, internal_identifier)
            index += 1

        return internal_identifier

    def _ping(self):
        for internal_identifier in set(self.status.keys()):
            status = self.status[internal_identifier]

            if status == Status.INITIAL:
                self.status[internal_identifier] = Status.SIMULATION
                self.process[internal_identifier] = self._start_simulation(internal_identifier)
                self.progress[internal_identifier] = 0
            elif status == Status.SIMULATION:
                return_code = self.process[internal_identifier].poll()

                if not return_code is None:
                    self.status[internal_identifier] = Status.ANALSYIS
                    self.process[internal_identifier] = self._start_analysis(internal_identifier)
                else:
                    iteration = self._get_iteration(internal_identifier)

                    if iteration > self.progress[internal_identifier]:
                        self.progress[internal_identifier] = iteration
                        logger.info("Running simulation %s ... (%d/%d iterations)" % (
                            internal_identifier, iteration,
                            self.simulation_parameters[internal_identifier]["iterations"]
                        ))

            elif status == Status.ANALSYIS:
                return_code = self.process[internal_identifier].poll()

                if not return_code is None:
                    self.status[internal_identifier] = Status.DONE

    def _start_simulation(self, internal_identifier):
        simulation_path = "%s/%s" % (self.working_directory, internal_identifier)
        parameters = self.simulation_parameters[internal_identifier]

        if not "iterations" in parameters:
            raise RuntimeError("Parameter 'iterations' must be set")

        if not "sample_size" in parameters:
            raise RuntimeError("Parameter 'sample_size' must be set (1pm, 1pct, 10pct, 25pct)")

        arguments = [
            parameters["java_path"], "-Xmx%s" % parameters["java_memory"],
            "-cp", parameters["class_path"],
            "ch.ethz.matsim.projects.astra_2018_002.RunASTRA2018002",
            "--config-path", parameters["config_path"].replace("{sample_size}", parameters["sample_size"]),
            "--config:controler.outputDirectory", "%s/output" % simulation_path,
            "--config:controler.lastIteration", str(parameters["iterations"]),
            "--config:global.numberOfThreads", str(parameters["number_of_threads"]),
            "--config:qsim.numberOfThreads", str(min(parameters["number_of_threads"], 12)),
            "--config:controler.writeEventsInterval", str(parameters["iterations"]),
            "--config:controler.writePlansInterval", str(parameters["iterations"]),
            "--config:global.randomSeed", str(parameters["random_seed"]),
            #"--model", "ZERO"
        ]

        scaling_factors = {
            "1pm": 0.001, "1pct": 0.01, "10pct": 0.1, "25pct": 0.25
        }

        arguments += ["--config:qsim.flowCapacityFactor", str(scaling_factors[parameters["sample_size"]])]
        arguments += ["--config:qsim.storageCapacityFactor", str(scaling_factors[parameters["sample_size"]])]

        if "config" in parameters:
            for option, value in parameters["config"].items():
                arguments += ["--config:%s" % option, str(value)]

        if "utilities" in parameters:
            for utility, value in parameters["utilities"].items():
                arguments += ["--utility:%s" % utility, str(value)]

        if "initial_identifier" in parameters:
            initial_internal_identifier = self.identifier_mapping[parameters["initial_identifier"]]
            initial_path = "%s/%s" % (self.working_directory, initial_internal_identifier)
            arguments += ["--config:plans.inputPlansFile", "%s/output/output_plans.xml.gz" % initial_path]

        stdout = open("%s/simulation_output.log" % simulation_path, "w+")
        stderr = open("%s/simulation_error.log" % simulation_path, "w+")

        return sp.Popen(arguments, stdout = stdout, stderr = stderr)

    def _start_analysis(self, internal_identifier):
        simulation_path = "%s/%s" % (self.working_directory, internal_identifier)
        parameters = self.simulation_parameters[internal_identifier]

        arguments = [
            parameters["java_path"], "-Xmx%s" % parameters["java_memory"],
            "-cp", parameters["class_path"],
            "ch.ethz.matsim.projects.astra_2018_002.analysis.trips.ConvertTripsFromEvents",
            "--network-path", "%s/output/output_network.xml.gz" % simulation_path,
            "--events-path", "%s/output/output_events.xml.gz" % simulation_path,
            "--network-path", "%s/output/output_network.xml.gz" % simulation_path,
            "--output-path", "%s/trips.csv" % simulation_path
        ]

        stdout = open("%s/analyis_output.log" % simulation_path, "w+")
        stderr = open("%s/analyis_error.log" % simulation_path, "w+")

        return sp.Popen(arguments, stdout = stdout, stderr = stderr)

    def run(self, identifier, parameters):
        internal_identifier = self._create_internal_identifier(identifier)
        simulation_path = "%s/%s" % (self.working_directory, internal_identifier)
        os.mkdir(simulation_path)

        self.identifier_mapping[identifier] = internal_identifier

        simulation_parameters = dict()
        simulation_parameters["random_seed"] = int(np.random.random() * 1e6)
        simulation_parameters.update(self.parameters)
        simulation_parameters.update(parameters)

        self.simulation_parameters[internal_identifier] = simulation_parameters
        self.status[internal_identifier] = Status.INITIAL

        self._ping()

    def is_running(self, identifier):
        self._ping()
        return self.status[self.identifier_mapping[identifier]] != Status.DONE

    def get_result(self, identifier):
        if self.is_running(identifier):
            raise RuntimeError("Cannot get result for %s, because it is still running" % identifier)

        internal_identifier = self.identifier_mapping[identifier]
        simulation_path = "%s/%s" % (self.working_directory, internal_identifier)

        return pd.read_csv("%s/trips.csv" % simulation_path, sep = ";")

    def _get_iteration(self, internal_identifier):
        scores_path = "%s/%s/output/scorestats.txt" % (self.working_directory, internal_identifier)

        if os.path.isfile(scores_path):
            with open(scores_path) as f:
                if len(f.read().split("\n")) < 2:
                    return 0

            df = pd.read_csv(scores_path, sep = "\t")

            if len(df) > 0:
                return df["ITERATION"].max()

        return 0

    def get_cost(self, identifier):
        internal_identifier = self.identifier_mapping[identifier]
        return self.simulation_parameters[internal_identifier]["iterations"]

    def cleanup(self, identifier):
        internal_identifier = self.identifier_mapping[identifier]
        simulation_path = "%s/%s" % (self.working_directory, internal_identifier)
        shutil.rmtree(simulation_path)

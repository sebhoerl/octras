import os, uuid, time, shutil
import subprocess as sp
import pandas as pd
import tqdm

class Status:
    PENDING = 0
    RUNNING = 1
    FAILED = 2
    SUCCESSFUL = 3

class Simulator:
    def __init__(self, parameters):
        self.parameters = {
            "java_path": "java",
            "java_memory": "10G",
            "number_of_parallel_runs": 1,
            "number_of_threads": 4
        }

        self.parameters.update(parameters)

        if not os.path.isdir(self.parameters["working_directory"]):
            raise RuntimeError("Working directory must exist: %s" % self.parameters["working_directory"])

        self.registry = {}

        self.scheduled = []
        self.running = {}
        self.processing = {}

    def _create_identifier(self):
        new_id = uuid.uuid4()

        while new_id in self.registry:
            new_id = uuid.uuid4()

        return new_id

    def schedule(self, parameters):
        identifier = self._create_identifier()
        path = os.path.abspath("%s/%s" % (self.parameters["working_directory"], identifier))

        self.registry[identifier] = dict(parameters = parameters, path = path, status = Status.PENDING)
        self.scheduled.append(identifier)

        return identifier

    def _start(self, identifier):
        simulation = self.registry[identifier]
        simulation["status"] = Status.RUNNING

        parameters = {}
        parameters.update(self.parameters)
        parameters.update(simulation["parameters"])

        if os.path.isdir(simulation["path"]):
            shutil.rmtree(simulation["path"])
        os.mkdir(simulation["path"])

        arguments = [
            parameters["java_path"], "-Xmx%s" % parameters["java_memory"],
            "-cp", parameters["class_path"],
            "ch.ethz.matsim.projects.astra_2018_002.RunASTRA2018002",
            "--config-path", parameters["config_path"],
            "--config:controler.outputDirectory", "%s/output" % simulation["path"],
            "--config:controler.lastIteration", str(parameters["iterations"]),
            "--config:global.numberOfThreads", str(parameters["number_of_threads"]),
            "--config:qsim.numberOfThreads", str(min(parameters["number_of_threads"], 12)),
            "--config:controler.writeEventsInterval", str(parameters["iterations"]),
            "--config:controler.writePlansInterval", str(parameters["iterations"]),
            "--model", "ZERO"
        ]

        if "config" in parameters:
            for option, value in parameters["config"].items():
                arguments += ["--config:%s" % option, str(value)]

        if "utilities" in parameters:
            for utility, value in parameters["utilities"].items():
                arguments += ["--utility:%s" % utility, str(value)]

        stdout = open("%s/run_output.log" % simulation["path"], "w+")
        stderr = open("%s/run_error.log" % simulation["path"], "w+")

        self.running[identifier] = sp.Popen(arguments, stdout = stdout, stderr = stderr)

    def _process(self, identifier):
        simulation = self.registry[identifier]

        parameters = {}
        parameters.update(self.parameters)
        parameters.update(simulation["parameters"])

        arguments = [
            parameters["java_path"], "-Xmx%s" % parameters["java_memory"],
            "-cp", parameters["class_path"],
            "ch.ethz.matsim.projects.astra_2018_002.analysis.trips.ConvertTripsFromEvents",
            "--network-path", "%s/output/output_network.xml.gz" % simulation["path"],
            "--events-path", "%s/output/output_events.xml.gz" % simulation["path"],
            "--network-path", "%s/output/output_network.xml.gz" % simulation["path"],
            "--output-path", "%s/trips.csv" % simulation["path"]
        ]

        stdout = open("%s/process_output.log" % simulation["path"], "w+")
        stderr = open("%s/process_error.log" % simulation["path"], "w+")

        self.processing[identifier] = sp.Popen(arguments, stdout = stdout, stderr = stderr)

    def ping(self):
        running_count = 0

        for identifier in list(self.running.keys()):
            process = self.running[identifier]
            return_code = process.poll()

            if not return_code is None:
                del self.running[identifier]

                if return_code == 0:
                    print("[Simulator] Run %s has finished. Sending to processing ..." % identifier)
                    self._process(identifier)
                else:
                    print("[Simulator] Run %s has failed." % identifier)
                    self.registry[identifier]["status"] = Status.FAILED
            else:
                running_count += 1

        for identifier in list(self.processing.keys()):
            process = self.processing[identifier]
            return_code = process.poll()

            if not return_code is None:
                del self.processing[identifier]

                if return_code == 0:
                    print("[Simulator] Run %s is processed." % identifier)
                    self.registry[identifier]["status"] = Status.SUCCESSFUL
                else:
                    print("[Simulator] Run %s could not be processed." % identifier)
                    self.registry[identifier]["status"] = Status.FAILED
            else:
                running_count += 1

        if running_count < self.parameters["number_of_parallel_runs"]:
            if len(self.scheduled) > 0:
                identifier = self.scheduled.pop(0)
                print("[Simulator] Run %s was started ..." % identifier)
                self._start(identifier)

    def wait(self, identifiers = None):
        self.ping()

        if identifiers is None:
            identifiers = set(self.running.keys()) | set(self.processing.keys()) | set(self.scheduled)

        pending = 0

        for identifier in identifiers:
            status = self.registry[identifier]["status"]

            if status == Status.PENDING or status == Status.RUNNING:
                pending += 1

        if pending > 0:
            total_iterations = 0

            for identifier in identifiers:
                total_iterations += self.registry[identifier]["parameters"]["iterations"]

            with tqdm.tqdm(total = total_iterations) as progress:
                pending = len(identifiers)
                previous_iterations = 0

                while pending > 0:
                    current_iterations = 0
                    pending = 0

                    for identifier in identifiers:
                        current_iterations += self.get_iteration(identifier)

                        status = self.registry[identifier]["status"]

                        if status == Status.PENDING or status == Status.RUNNING:
                            pending += 1

                    progress.update(current_iterations - previous_iterations)
                    previous_iterations = current_iterations

                    time.sleep(1)
                    self.ping()

    def get_iteration(self, identifier):
        scores_path = "%s/output/scorestats.txt" % self.registry[identifier]["path"]

        if os.path.isfile(scores_path):
            with open(scores_path) as f:
                if len(f.read().split("\n")) < 2:
                    return 0

            df = pd.read_csv(scores_path, sep = "\t")

            if len(df) > 0:
                return df["ITERATION"].max()

        return 0

    def get(self, identifier):
        self.wait([identifier])
        simulation = self.registry[identifier]
        return pd.read_csv("%s/trips.csv" % simulation["path"], sep = ";")

    def cleanup(self, identifier):
        simulation = self.registry[identifier]
        shutil.rmtree(simulation["path"])

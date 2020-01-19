# MATSim use case

The following represents a use case for the optimization/calibration framework
for the agent-based transport simulator MATSim. To run the use case and use
the latest version of `octras`, it is best to create a local virtual environment.

## Setting up the environment

```sh
virtualenv matsim_use_case
```

Enter the environment (and do this everything you go back to the use case) ...:

```sh
source matsim_use_case/bin/activate
```

... and install `octras`:

```sh
pip3 install -e /path/to/octras/repository
```

## Setting up the data

There are currently to use cases:

- [Zurich](https://polybox.ethz.ch/index.php/s/BTtgWYke7F7WPLg) (only accessible for ETH Zurich, password required)
- Paris (publicly available, TODO!!!)

You can download the data by clicking the link above and unpacking the archive
to your hard drive. It contains a runnable MATSim scenario and the respective
data sets. In order to run the simulation, you will need to have installed a
current version of the Java Runtime Environment, at least of version 8.

## Running the calibration

Now go to the cloned `octras` repository and to the folder `use_caes`. In this
folder you find a run file `run.py` and an example configuration file
`example.yml`, which controls the calibration.

In `example.yml` you'll find three paths, which must be adjusted:

- `simulation_path` should contain the path to the simulation data that you have downloaded.
- `working_directory` should point to an existing (preferably empty) directory in which the
runner will save all the temporary simulation data while running.
- `calibration.output_path` should point to a location of a `pickle` file with extension
`*.p`, which tracks the calibration process and provides all information to perform
analyses of the algorithms.

The `example.yml` describes some of the options that are available for the use
case. Note the `optimization` section. One of them is commented out and represents
a SPSA use case, while the other one defines a Bayesian Optimization use case
with the MES algorithm.

Once the paths are adjusted, one can run:

```sh
python3 run.py example.yml
```

The progress of the calibration can be followed using some of the available
plotting scripts, for instance:

```sh
python3 plotting/plot_problem.py /path/to/output.p
```

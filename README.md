# Optimization and Calibration for Transport Simulation

The `octras` package makes it easy to run optimization and calibration procedures
for large-scale transport simulations. It offers a couple of tools for easy analysis
of the simulation runs and interfaces to put simulators like MATSim into an
optimization loop.

Implemented optimization / calibration algorithms:

- **Random walk**
- **[FDSA][1]**: Finite Difference Stochastic Approximation
- **[SPSA][2]**: Simultaneous Perturbation Stochastic Approximation
- **[Opdyts][3]**: *Flötteröd, G. (2017) A search acceleration method for optimization problems with transport simulation constraints, Transportation Research Part B, 98, 239-260.*
- **[CMA-ES][4]**: Covariance Matrix Adaptation Evolution Strategy
- **[scipy.optimize][5]**: All algorithms contained in the `scipy.optimize` package can be used.

# Installing the tools

We provide a conda environment in `environment.yml` which lists all the necessary dependencies. Once set up, you can call

```sh
conda develop src
```

in the main directory of the cloned project to let conda know about the `octras` package which is located in `/src`. Afterwards, you'll be able to use `octras` inside of this enviroment, for instance, in a Jupyter notebook.

# Examples

Currently, there is one example on running a simulation for Corsica using the eqasim extension of MATSim. In the example, the mode-specific constant of a discrete choice model is calibrated to achieve a certain mode share for the `car` mode in the simulation.

To run the example, you'll need Jupyter. You can install it easily in your environment where you have also installed `octras` by

`conda install jupyter matplotlib`

and then starting it via

`jupyter`

and then selecting the notebook in `examples`. You'll need a couple of knowledge on how to build MATSim / Java packaged. However, if you have certain standard tools installed (such as Maven and Git), the notebook will guide you through the process.

# Running the tests

To verify that the framework works as expected, enter your `octras` conda environment and execute:

```sh
python3 -m pytest tests
```
The unit tests should start running then and all of them should finish without error.

[1]: https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
[2]: https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
[3]: https://www.sciencedirect.com/science/article/pii/S0191261516302466
[4]: https://en.wikipedia.org/wiki/CMA-ES
[5]: https://docs.scipy.org/doc/scipy/reference/optimize.html

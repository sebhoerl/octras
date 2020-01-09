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

# Running with MATSim

For reference, there is a [use case](use_case/README.md) for MATSim that uses the framework.

[1]: https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
[2]: https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
[3]: https://www.sciencedirect.com/science/article/pii/S0191261516302466
[4]: https://en.wikipedia.org/wiki/CMA-ES
[5]: https://docs.scipy.org/doc/scipy/reference/optimize.html

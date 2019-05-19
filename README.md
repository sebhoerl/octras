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

A current scenario (internal for ETH Zurich) is located at:
https://polybox.ethz.ch/index.php/s/Bh058pctP90ck0D

The corresponding scripts can be found in the `use_case` folder. In the future,
components from there should be fed back as features to the overall `octras`
package.

To run an optimization, the data linked above should be copied to the directory
`simulation` inside of the `use_case` directory. Also, a directory `use_case/temp`
should be created. Then, one can run:

```bash
PYTHONPATH=/path/to/repository_directory python3 run_use_case.py --problem mode_share --algorithm cma_es --log_path temp/log.p
```

Multiple other options are available in the runs script (just run it without parameters).

The `log.p` can be analysed with the following general script:

```bash
python3 plotting/plot_problem.py temp/log.p
```

or with custom scripts to get insights into the dynamics of specific algorithms:

```bash
python3 plotting/plot_spsa.py temp/log.p
```
```bash
python3 plotting/plot_opdyts.py temp/log.p
```

The latter will show, for instance, the uniformity gap and equlibirum gap that
is used internall in *opdyts*.

[1]: https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
[2]: https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
[3]: https://www.sciencedirect.com/science/article/pii/S0191261516302466
[4]: https://en.wikipedia.org/wiki/CMA-ES
[5]: https://docs.scipy.org/doc/scipy/reference/optimize.html

# MATSim use case

A current scenario (internal for ETH Zurich) is located at:
https://polybox.ethz.ch/index.php/s/Bh058pctP90ck0D

The corresponding scripts can be found in the `use_case` folder. In the future,
components from there should be fed back as features to the overall `octras`
package.

To run an optimization, the data linked above should be copied to the directory
`simulation` inside of the `use_case` directory. Also, a directory `use_case/temp`
should be created. Then, one can run:

```bash
PYTHONPATH=/path/to/repository_directory python3 run_use_case.py example.yml
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

# Optimization/Calibration for MATSim

For large-scale transport simulations such as MATSim calibration and optimization
are important topics. *Calibration* is needed when a certain set of reference
values are available (such as mode shares, links counts, ...) and *optimization*
is important when policy decisions should be evaluated (such as finding an
optimal road pricing scheme, fleet mix for a taxi operator, ...).

The general setup is that a number of input parameters (*u*) need to be chosen
such that the system arrives at a certain state (*x*). This state *x* is in some
sense optimal, i.e. it minimizes a given objective function *J(x)*.

A concrete example would be as follows: Let *u* be all choice parameters that go
into a MATSim model and let *x* be the measured mode shares in the simulation. Also,
we have reference values *r* for those mode shares available. We then want to find
an optimal *u** where *J(x; r) = (x - r)**2* is minimal. This will bring the
simulated mode shares as close as possible to the refrences values by adjusting
the choice parameters of the model.




















##

class Problem:
    """
        This class defines a optimization/calibration problem. A number
        of attributes should be defined:

        self.number_of_parameters = ?
        self.number_of_states = ?

        Additional information can be provided:

        self.information = {}
        self.reference_state = ?
    """

    def __init__(self, car_reference, pt_reference):
        raise NotImplementedError()

    def prepare(self, x):
        """
            This function gets a vector of numeric parameters. The task is
            to return a dictionary with instructions for the simulator. The
            return format is either just a dictionary or a tuple (dictionary, cost)
            with cost being the execution cost of the run.
        """
        raise NotImplementedError()

    def evaluate(self, x, result):
        """
            This is the objective function. It receives the numeric parameters
            as the first argument and the output of the simulator as the
            second argument. It should either return only an objective value
            or a tuple of (objective value, state).
        """
        raise NotImplementedError()

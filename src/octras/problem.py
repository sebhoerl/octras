class Problem:
    """
        This is the basic interface for a problem in octras. See the detailed
        description of the methods below that need to be implemented.
    """

    def get_information():
        """
            This method is supposed to return a dictionary with general information
            on the optimization problem. These can be arbitrary values, but at least
            the dictionary must contain one field `number_of_parameters`.
        """
        raise NotImplementedError()

    def parameterize(self, x):
        """
            This method receives a vector of numeric parameters. The task is to return
            a dictionary with instructions for the simulator. Which fields are required
            depends on the specific simulator. The return format is either only a dictionary
            or a tuple of the dictionary and the associated cost of the run.
        """
        raise NotImplementedError()

    def evaluate(self, x, response):
        """
            This method represents the objective function. It receives the numeric
            parameters as the first argument and the response of the simulator as the
            second argument. The method should either return only a numeric objective
            value or a tuple of (objective_value, state) where state is an arbitrary
            dictionary which might contain additional information for logging or informing
            the optimization algorithm.
        """
        raise NotImplementedError()

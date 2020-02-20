class Simulator:
    """ Defines a simulator with custom parameters. """

    def run(self, identifier, parameters):
        """
            Starts a new simulation in background with a certain idenfier.
            The parameters are a dictionary with information about the run.
        """
        raise NotImplementedError()

    def ready(self, identifier):
        """
            Returns Boolean whether the run with certain identifier has finished.
        """
        raise NotImplementedError()

    def get(self, identifier):
        """
            Gets the result for the run with the identifier. Should raise an
            error if the simulation is not done yet.
        """
        raise NotImplementedError()

    def clean(self, identifier):
        """
            Cleans the result for the run with the identifier. Should raise
            an error if the simulation is not done yet.
        """
        raise NotImplementedError()

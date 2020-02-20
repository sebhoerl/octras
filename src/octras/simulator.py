class Simulator:
    def run(self, identifier, parameters):
        raise NotImplementedError()

    def ready(self, identifier, parameters):
        raise NotImplementedError()

    def get(self, identifier):
        raise NotImplementedError()

    def clean(self, identifier):
        raise NotImplementedError()

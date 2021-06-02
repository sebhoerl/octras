from octras import Simulator

class GenericTestSimulator(Simulator):
    __test__ = False

    def __init__(self):
        self.results = {}

    def ready(self, identifier):
        return True

    def get(self, identifier):
        return self.results[identifier]

    def clean(self, identifier):
        del self.results[identifier]

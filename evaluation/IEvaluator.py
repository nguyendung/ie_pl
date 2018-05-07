from abc import ABCMeta, abstractmethod


class IEvaluator(metaclass=ABCMeta):
    def __init__(self):
        self.predicteds = []
        self.actuals = []
        pass

    @abstractmethod
    def measure(self):
        pass

    def set_predicted_values(self, predicteds):
        self.predicteds = predicteds

    def set_actual_values(self, actuals):
        self.actuals = actuals





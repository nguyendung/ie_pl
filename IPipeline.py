from abc import ABCMeta, abstractmethod


class IPipeline(metaclass=ABCMeta):
    def __init__(self):
        self.description = ""
        self.available_modules = {}
        self.running_modules = []
        pass

    '''
    Configuration will include
    1. Static configuration: module name, module api, version
    2. Dynamic: runtime configuration
    '''
    @abstractmethod
    def load_module_configuration(self):
        pass

    @abstractmethod
    def load_running_configuration(self):
        pass

    @abstractmethod
    def get_result(self):
        pass

    @abstractmethod
    def clear_pipeline(self):
        pass

    @abstractmethod
    def run_pipeline(self, img):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def print_pipeline_info(self):
        pass

    @abstractmethod
    def self_check(self):
        pass
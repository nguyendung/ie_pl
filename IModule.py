from abc import ABCMeta, abstractmethod


class IModule(metaclass=ABCMeta):
    def __init__(self):
        self.mo_code = ""
        self.api_url = ""
        self.output_folder = ""
        self.option = ""
        self.output_data = []

    def set_module_config(self, mo_code, api_url, output_folder, option):
        self.mo_code = mo_code
        self.api_url = api_url
        self.output_folder = output_folder
        self.option = option
        self.output_data.clear()

    @abstractmethod
    def run_module(self, input_data):
        print("Running {} with api: {}".format(self.mo_code, self.api_url))
        pass

    @abstractmethod
    def write_output(self):
        pass

    # config includes output folder and api_url
    @abstractmethod
    def load_config(self):
        pass

    def get_output(self):
        return self.output_data
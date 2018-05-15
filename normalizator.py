from IModule import IModule
import requests
import cv2
from flask import Response, make_response
import numpy as np
from os.path import join


class Normalizator(IModule):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(Normalizator, cls).__new__(cls)
            cls.__instance.__initialized = False
        return cls.__instance

    def __init__(self):
        if self.__initialized: return
        self.__initialized = True
        super().__init__()

    def run_module(self, input_data):
        super().run_module(input_data)

        for img in input_data:
            content_type = 'image/jpeg'
            headers = {'content-type': content_type}
            _, img_encoded = cv2.imencode('.png', img)
            res = requests.post(self.api_url + self.option, data=img_encoded.tostring(), headers=headers)

            np_arr = np.fromstring(res.content, np.uint8)
            res_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            self.output_data.append(res_img)

    def write_output(self):
        img_index = 1
        for img in self.output_data:
            cv2.imwrite(join(self.output_folder, "{}-{}.png".format(self.mo_code, img_index)), img)
            img_index += 1

    def load_config(self):
        pass

from IModule import IModule
import requests
import cv2
from flask import Response, make_response
import numpy as np
from os.path import join
import json
import unicodedata
from define import TEXT_OUTPUT


class OcrEngine(IModule):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(OcrEngine, cls).__new__(cls)
            cls.__instance.__initialized = False
        return cls.__instance

    def __init__(self):
        if self.__initialized: return
        self.__initialized = True
        super().__init__()

    def run_module(self, input_data):
        super().run_module(input_data)

        for img in input_data:
            _, img_encoded = cv2.imencode('.png', img)
            payload = {"image": img_encoded.tobytes()}
            res = requests.post(self.api_url + self.option, files=payload).json()

            ocr_text = unicodedata.normalize("NFKC", res["prediction"]) if res["success"] else ""

            self.output_data.append(ocr_text)

    def write_output(self):
        with open(join(self.output_folder, TEXT_OUTPUT), 'w') as out_file:

            out_file.write("{}\t{}\n".format('id', 'text'))

            text_count = 1
            for text in self.output_data:
                out_file.write("{}\t{}\n".format(str(text_count), text))
                text_count += 1
        out_file.close()

    def load_config(self):
        pass

from IModule import IModule
import requests
import cv2
import numpy as np
from os.path import join
from requests_toolbelt import MultipartDecoder
import base64
from define import TEXT_OUTPUT, IMG_OUTPUT, VIA_FILE
import codecs


class Segmentator(IModule):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(Segmentator, cls).__new__(cls)
            cls.__instance.__initialized = False
        return cls.__instance

    def __init__(self):

        if self.__initialized: 
            self.boxes.clear()
            self.debug_img = None
            self.via_data = None
            return
        self.__initialized = True
        super().__init__()
        self.boxes = []
        self.debug_img = None
        self.via_data = None

    def run_module(self, input_data):
        super().run_module(input_data)

        for img in input_data:
            content_type = 'image/jpeg'
            headers = {'content-type': content_type}

            _, img_encoded = cv2.imencode('.png', img)
            res = requests.post(self.api_url + self.option, data=img_encoded.tostring(), headers=headers)
            decoder = MultipartDecoder.from_response(res)
            for part in decoder.parts:

                box_location, via_data_check, debug_img_check = self.get_box_location_from_message_header(part)
                if box_location is not None:
                    self.boxes.append(box_location)
                    np_arr = np.fromstring(base64.b64decode(part.text), np.uint8)
                    res_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    self.output_data.append(res_img)
                else:
                    if debug_img_check is True:
                        np_arr = np.fromstring(base64.b64decode(part.text), np.uint8)
                        res_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        self.debug_img = res_img
                    if via_data_check is True:
                        np_arr = str(part.text)
                        self.via_data = np_arr

    def get_box_location_from_message_header(self, part):
        k, v = list(part.headers.items())[0]
        name_box = v.decode('utf-8').split(';')[1].strip()
        name, box_location = name_box.split("=")
        if "debug_img" not in box_location:
            if "via_data" not in box_location:
                return box_location[1:-1], None, None
            else:
                return None, True, None
        else:
            return None, None, True

    def write_output(self):
        img_index = 1
        for img in self.output_data:
            cv2.imwrite(join(self.output_folder, "{}-{}.png".format(self.mo_code, img_index)), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            img_index += 1

        with open(join(self.output_folder, TEXT_OUTPUT), 'w') as out_file:
            out_file.write("{}\t{}\n".format('id', 'box'))
            text_count = 1
            for text in self.boxes:
                out_file.write("{}\t{}\n".format(str(text_count), text))
                text_count += 1
        out_file.close()

        cv2.imwrite(join(self.output_folder, IMG_OUTPUT), self.debug_img)

        with open(join(self.output_folder, VIA_FILE), 'w') as out_file:
            for text in self.via_data.split("\n"):
                out_file.write("{}\r\n".format(text))
        out_file.close()

    def load_config(self):
        pass

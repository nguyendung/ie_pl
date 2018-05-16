from flask import Flask, request
import flask
from base_pipeline import BasePipeline
from flask import g
import numpy as np
import cv2
import base64
import io
import pandas as pd
from multiprocessing import Value
import sys
from define import DEBUG_FOLDER, VIA_FILE, TEXT_OUTPUT, IMG_OUTPUT
from os.path import join
import base64

app = Flask(__name__)
res_cache = {}
module_cache = {}


@app.route("/process", methods=["POST"])
def predict():
    data = {"success": True}

    if request.method == "POST":

        np_arr = np.fromstring(request.files["image"].read(), np.uint8)
        img_size = sys.getsizeof(np_arr) - 96
        res_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        uuid_string = "".join(map(chr, request.files['uuid'].read()))

        label = None

        if 'label' in request.files:
            label_string = base64.b64decode(request.files['label'].read()).decode('utf8')
            label = pd.read_csv(io.StringIO(label_string), sep="\t")

        img_name = request.files['img_name'].read().decode('utf8')

        pipeline = BasePipeline()
        running_cans = pipeline.get_all_running_configs()
        config_index = 0
        for [des, a_running_config] in running_cans:
            if config_index not in module_cache:
                module_cache[config_index] = des

            cer, out_folder = pipeline.run_pipeline(res_img, running_config=a_running_config, img_size=img_size, img_name=img_name, labels=label)

            if uuid_string not in res_cache:
                res_cache[uuid_string] = {}
            if config_index not in res_cache[uuid_string]:
                res_cache[uuid_string][config_index] = []

            res_cache[uuid_string][config_index].append([cer, out_folder, img_name])
            config_index += 1

    return flask.jsonify(data)


@app.route("/download", methods=["POST"])
def get():
    data = {}
    uuid_string = "".join(map(chr, request.files['uuid'].read()))
    print(res_cache)
    
    if uuid_string in res_cache:
        pl_data = res_cache[uuid_string]
        for key in list(pl_data.keys()):
            if str(key) not in data:
                data[str(key)] = {}

            data[str(key)]["des"] = module_cache[key]

            via_data = []
            via_data.append("{}\r\n".format('#filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes'))
            sum_cer = 0.0
            count_img = 0
            img_data = {}
            ocr_text = {}
            cer_value = {}
            for [cer, out_folder, img_name] in pl_data[key]:
                sum_cer += cer
                count_img += 1

                cer_value[img_name] = str(cer)

                # Get VIA Data
                with open(join(out_folder, DEBUG_FOLDER, VIA_FILE), 'r') as f:
                    contents = f.readlines()

                for c in contents:
                    via_data.append(c)

                # Get Image
                img = cv2.imread(join(out_folder, DEBUG_FOLDER, IMG_OUTPUT))
                _, img_encoded = cv2.imencode(".png", img)
                img_data[img_name] = base64.b64encode(img_encoded).decode('utf-8')

                # Get OCR Text
                with open(join(out_folder, DEBUG_FOLDER, TEXT_OUTPUT), 'r') as f:
                    contents = f.readlines()
                ocr_text[img_name] = contents

            via_data_str = ''.join(i for i in via_data)

            data[str(key)]["ned"] = sum_cer/count_img
            data[str(key)]["img_count"] = count_img
            data[str(key)]["via_data"] = via_data_str
            data[str(key)]["imgs"] = img_data
            data[str(key)]["ocr"] = ocr_text
            data[str(key)]["cer"] = cer_value
    else:
        data["Success"] = False

    return flask.jsonify(data)


if __name__ == '__main__':

    app.run(port=5006, host="0.0.0.0")


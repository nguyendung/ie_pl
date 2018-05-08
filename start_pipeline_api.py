from flask import Flask, request
import flask
from base_pipeline import BasePipeline
from flask import g
import numpy as np
import cv2
import base64
import io
import pandas as pd


app = Flask(__name__)


@app.route("/process", methods=["POST"])
def predict():
    data = {"success": False}
    if getattr(g, "_state", None) == "Busy":
        data["reason"] = "Not support multiple requests at the momment"
        return flask.jsonify(data)
    else:
        setattr(g, "_state", "Busy")

    data = {"success": True}

    if request.method == "POST":
        nparr = np.fromstring(flask.request.files["image"].read(), np.uint8)
        res_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        label_string = base64.b64decode(request.files['label'].read()).decode('utf8')
        label = pd.read_csv(io.StringIO(label_string), sep="\t")
        pipeline = BasePipeline()
        pipeline.run_pipeline(res_img, img_name="test.png", labels=label)

        # if request.files.get("image"):
        #
        #     # read the image in PIL format
        #     image = request.files["image"].read()
        #     image = Image.open(io.BytesIO(image))
        #
        #     # run pipeline and get result
        #     pipeline = Pipeline()
        #     res = pipeline.img_process(image, int(request.args["type"]))
        #
        #     if res:
        #         data = {}
        #         data["prediction"] = pipeline.get_pipeline_final_result()
        #         data["images"] = pipeline.get_pipeline_module_result('se')
        #         data["success"] = True
        #     else:
        #         data = {"success": False}
        #         data["reason"] = "Pipeline Failure"

    setattr(g, "_state", "Free")
    return flask.jsonify(data)


if __name__ == '__main__':
    app.run(port=6000, host="0.0.0.0")
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
        cer = pipeline.run_pipeline(res_img, labels=label)
        data["cer"] = cer

    setattr(g, "_state", "Free")
    return flask.jsonify(data)


if __name__ == '__main__':
    app.run(port=6000, host="0.0.0.0")
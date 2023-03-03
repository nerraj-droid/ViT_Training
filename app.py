import os.path

from flask import Flask, request, jsonify
from transformers import ViTForImageClassification, ViTImageProcessor
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'topredict'

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['TO_PREDICT_FOLDER'] = TO_PREDICT_FOLDER


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['img_path']
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        path = "./out/MyKvasirV2Model/20_2023-03-01-14-45-39/model"
        model = ViTForImageClassification.from_pretrained(path)
        feature_extractor = ViTImageProcessor.from_pretrained(path)
        classifier = VisionClassifierInference(
            feature_extractor,
            model,
        )
        result = classifier.predict(filepath)
    else:
        result = 'Error'
    return jsonify({'result': result})

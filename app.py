import os.path

import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference
from flask import Flask, request
from PIL import Image

UPLOAD_FOLDER = 'topredict'

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# app.config['TO_PREDICT_FOLDER'] = TO_PREDICT_FOLDER
def is_valid_image(file_path):
    try:
        # open image file
        with Image.open(file_path) as img:
            # return True if the image can be opened
            return True
    except:
        # if there is an error, return False
        return False


def predict_mango_disease(filepath, model_path):
    # Load the pre-trained model and image processor
    model = ViTForImageClassification.from_pretrained(model_path)
    processor = ViTImageProcessor.from_pretrained(model_path)
    classifier = VisionClassifierInference(
        processor,
        model,
    )

    result = classifier.predict(filepath)
    img = Image.open(filepath)
    # Transform the image
    encoding = processor(images=img, return_tensors="pt")

    # Predict and get the corresponding label identifier
    pred = model(encoding['pixel_values'])

    # Get label index
    predicted_class_idx = pred.logits.argmax(-1).tolist()[0]
    softmax = torch.nn.Softmax(dim=0)

    pred_soft = softmax(pred[0][0])
    pred_soft = torch.mul(pred_soft, 100)
    probabilities_list = pred_soft.tolist()
    probabilities = classifier.predict_image(img)
    confidence = probabilities[predicted_class_idx]
    print(f"index: {predicted_class_idx}")
    print(f"probability: {probabilities}")
    print(f"confidence: {confidence}")
    print(f"list_prob: {probabilities_list}")

    if any(val >= 95 for val in probabilities_list):
        disease_label = {'predicted_label': result}
    else:
        disease_label = {'predicted_label': 'Unknown: Not in the DATASET'}

    return disease_label


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['img_path']
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        model_path = "./out/MyKvasirV2Model/20_2023-03-01-14-45-39/model"

        result = predict_mango_disease(filepath, model_path)
    return result

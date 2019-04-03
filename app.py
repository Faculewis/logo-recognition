import base64
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template
from flask_cors import CORS
from keras.models import load_model
from keras.preprocessing import image
from io import BytesIO
import tensorflow as tf


app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
model = None
graph = None
model_segmentation = None
graph_segmentation = None


@app.route('/')
def index():
    return render_template('index.html', name='Logos')


@app.route('/predict', methods=['POST'])
def predict():
    base_img = request.form['img']

    x = prepare_image(base_img, (160, 160))
    # classification_data = predict_classification(x)
    segmentation_img = predict_segmentation(x)

    result = {}
    if segmentation_img is not None: # and classification_data is not None:
        result = {
            # "category": classification_data[0],
            # "confidence": classification_data[1],
            "image": segmentation_img
        }
    return jsonify(result)


def prepare_image(base64_img, target_size=(160, 160)):
    img = BytesIO(base64.b64decode(base64_img.replace('data:image/jpeg;base64,', '')))
    img = image.img_to_array(image.load_img(img, target_size=target_size)) / 255.
    x = np.expand_dims(img, axis=0)
    return x


def predict_segmentation(x):
    global model_segmentation

    with graph_segmentation.as_default():
        predicted = model_segmentation.predict(x, verbose=1)
        img = image.array_to_img(predicted[0])
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        base_img = base64.b64encode(img_byte_arr)
        return base_img.decode("utf-8")


def predict_classification(x):
    global model

    with graph.as_default():
        predicted = model.predict(x, verbose=1)
        idx = np.argmax(predicted[0])
        category = idx.item()
        confidence = predicted[0][idx].item()
        return [category, confidence]


def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def load_vgg_model():
    global model
    global graph
    model = load_model('vgg16_32_160x160.h5')
    graph = tf.get_default_graph()
    print("Model loaded")


def load_unet_model():
    global model_segmentation
    global graph_segmentation
    model_segmentation = tf.keras.models.load_model('UNet_32.h5', {
        "bce_dice_loss": bce_dice_loss,
        "dice_loss": dice_loss
    })
    graph_segmentation = tf.get_default_graph()
    print("Model Segmetation loaded")


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_vgg_model()
    load_unet_model()
    app.run()

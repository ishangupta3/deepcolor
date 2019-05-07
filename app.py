import flask
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.io import imsave
import scipy.misc
from scipy import ndimage, misc
from PIL import Image
import io
from io import BytesIO, StringIO
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from flask import send_file, abort
import pandas as pd



# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

def init():
    global model,graph
    # load json and create model
    json_file = open('mastermodel/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('mastermodel/model.h5')
    print("Loaded model from disk")
    # evaluate loaded model on test data
    model.compile(optimizer='rmsprop', loss='mse')
    graph = tf.get_default_graph()


# Cross origin support
def sendResponse(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response


def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')



# API for prediction
@app.route("/predict", methods=["POST"])
def predict():


    final_image = None

    try:
        if flask.request.method == "POST":
            if flask.request.files.get("image"):
                image = flask.request.files["image"].read()
                image = Image.open(io.BytesIO(image))
                image = img_to_array(image)
                image = scipy.misc.imresize(image, (64, 64))
                sonnan_X = rgb2lab(1.0 / 255 * image)[:, :, 0]
                sonnan_Y = rgb2lab(1.0 / 255 * image)[:, :, 1:]
                sonnan_Y = sonnan_Y / 128
                sonnan_X = sonnan_X.reshape(1, 64, 64, 1)
                sonnan_Y = sonnan_Y.reshape(1, 64, 64, 2)

                with graph.as_default():
                    output =   model.predict(sonnan_X)
                    output = output * 128
                    canvas = np.zeros((64, 64, 3))
                    canvas[:, :, 0] = sonnan_X[0][:, :, 0]
                    canvas[:, :, 1:] = output[0]




                    imsave("img_resultBOOM.png", lab2rgb(canvas))
                    imsave("img_gray_scaleBOOM.png", rgb2gray(lab2rgb(canvas)))


                    final_image = lab2rgb(canvas)  # converted image from LAB

    except:
        return abort(404, description="Incorrect Image sent through bro")

    if final_image is not None:
        im = Image.fromarray(np.uint8(final_image * 255))
        return serve_pil_image(im)

    else:
        return abort(404, description="Incorrect Image sent through bro")









# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
"please wait until server has fully started"))
    init()
    app.run(host='0.0.0.0',threaded=True)


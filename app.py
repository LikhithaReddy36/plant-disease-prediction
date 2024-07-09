import division, print_function

from future
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf
from os.path import join, dirname, realpath

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(_name

# Model saved with Keras model.save()
MODEL_PATH ='inceptionV3.h5'

# Loading trained model
model = load_model(MODEL_PATH)
def model_predict(img_path, model):
print(img_path)
img = image.load_img(img_path, target_size=(224, 224))

# Preprocessing the image
x = image.img_to_array(img)
# x = np.true_divide(x, 255)
## Scaling
X=X/255
x = np.expand_dims(x, axis=0)

preds = model.predict(x)
preds=np.argmax(preds, axis=1)
print(preds)
if preds == 0:
preds="Tomato_Early_blight"
elif preds == 1:
preds="Tomato_Late_blight"
elif preds == 2:
preds="Tomato_Leaf_Mold"
elif preds == 3:
preds="Tomato_Mosaic_Virus"
elif preds == 4:
preds="Tomato_Healthy"

return preds

@app.route('/', methods=['GET'])
def index():
# Main page
return render_template('index.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
if request.method mm 'POST':
# Get the file from post request

img = request.files['file']
img_path = "static/uploads/" + img.filename
img. save(img_path)

# Make prediction
preds = model_predict(img_path, model)
solution = solutions(preds)
ans=f'{preds} |I|| SOLUTIONS:\n {solution}"
return ans
return None

Tomato_Early_blight = =##

1. Mancozeb Flowable with Zinc Fungicide Concentrate
2. Spectracide Imunox Multi-Purpose Fungicide Spray Concentrate For Gardens
3. Southern Ag - Liquid Copper Fungicide
4. Bonide 811 Copper 4E Fungicide
S. Daconil Fungicide Concentrate.

Tomato_Healthy = "##
\nYour Plant Is Healthier.\m

Tomato_Late_blight = "*#

Plant resistant cultivars when available.
Remove volunteers from the garden prior to planting and space plants far enough apart to allow for plenty of air circulation.
Water in the early morning hours, or use soaker hoses, to give plants time to dry out during the day - avoid overhead irrigation.
Destroy all tomato and potato debris after harvest.

An

if

Tomato_Leaf_Mold m """
\nFungicides :
1. Difenoconazole and Cyprodinil
2. Difenoconazole and Mandipropamid
3. Cymoxanil and Famoxadone
4. Azoxystrobin and Difenoconazole

Tomato_mosaic_virus = """
\n
Fungicides will not treat this viral disease.
Avoid working in the garden during damp conditions (viruses are easily spread when plants are wet).
Frequently wash your hands and disinfect garden tools, stakes, ties, pots, greenhouse benches, etc.
Remove and destroy all infected plants.Do not compost.
Do not save seed from infected crops.
...

def solutions(disease):
switcher = {
"Tomato_Early_blight": Tomato_Early_blight ,
"Tomato_healthy": Tomato_Healthy ,
"Tomato_Late_blight" : Tomato_Late_blight,
"Tomato_Leaf_Mold" : Tomato_Leaf_Mold,
"Tomato_Tomato_mosaic_virus" : Tomato_mosaic_virus,
}
return switcher.get(disease,"Not Found In The List")

if __name__== " main_":

app.run(port=5001,debug=True)


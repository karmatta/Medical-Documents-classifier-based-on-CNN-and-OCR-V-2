seed = 1234
import pandas as pd
import os, glob
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)
# fix random seed for reproducibility
import tensorflow as tf
tf.compat.v1.random.set_random_seed(seed)
import keras
from numpy import array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import text_utilities as tu
import modeling_utils as mu
import image_utilities as iu
import ocr
from keras_self_attention import SeqSelfAttention
import doc_classifier_model as dcm
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
#from keras.optimizers import Adagrad
from keras import backend as K
import json
import sys
import time
import operator
from PIL import Image
import base64
from PIL import Image
from io import BytesIO
from keras.applications import ResNet50
from flask import Flask, request
import requests
import logging


sess_config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
allow_soft_placement=True, device_count = {'CPU': 1})
sess = tf.Session(graph=tf.get_default_graph(),config=sess_config)
K.set_session(sess)


with open('model_config', 'r') as configfile:
    config = json.load(configfile)
    
def start_timer():
    request.start_time = time.time()

def stop_timer(response):
    resp_time = time.time() - request.start_time
    sys.stderr.write("Response time: %ss\n" % resp_time)
    return response

def record_request_data(response):
    sys.stderr.write("Request path: %s Request method: %s Response status: %s\n" % (request.path, request.method, response.status_code))
    return response

def setup_metrics(app):
    app.before_request(start_timer)
    # The order here matters since we want stop_timer
    # to be executed first
    app.after_request(record_request_data)
    app.after_request(stop_timer)
    
app = Flask(__name__)
setup_metrics(app)

sess_config = tf.ConfigProto()

class LoadModel:

    def __init__(self, model_path):
        self.sess  = tf.Session(config=sess_config)
        K.set_session(self.sess) 
        print('Loading ResNet50...')
        self.resnet50 = ResNet50(weights='imagenet', pooling=max, include_top=False)
        print('ResNet50 Loaded...')
        print('Loading Doc Classifier...')
        self.model = dcm.build_doc_classifier(config['input_text_shape'], config['input_img_shape'], config['n_classes'], config['vocab_size'])
        self.model.load_weights(model_path)
        print('Doc Classifier Loaded...')
        self.graph = tf.get_default_graph()

    # Function to decode image to jpg from base64
    def decode_image(self, jsn):
        self.data = json.loads(json.loads(jsn))
        self.imageid = self.data['imageId']
        print("Image class:", self.imageid)
        return Image.open(BytesIO(base64.b64decode(self.data['base64ImageData'][0]))), self.imageid
    
    
    # Function to generate resnet features
    def generate_image_features(self, image, reshape=(225, 225)):
        # GENERATING FEATURES
        self.image = image.resize(reshape)
        self.x = np.asarray(self.image, dtype="int32" )
        self.x = np.expand_dims(self.x, axis=0) 
        with self.graph.as_default():
            self.img_features = self.resnet50.predict(self.x, verbose=0).squeeze().flatten()
        return self.img_features

    # Predict function
    def predict(self, jsn):
    
        self.start = time.time()
        self.start_json = time.time()
        self.img, self.imageid = self.decode_image(jsn)
        self.end_json = time.time()

        # Create text features
        self.start_clean_string = time.time()
        self.x_txt = tu.clean_string(ocr.get_text_from_image(self.img))
        self.end_clean_string = time.time()

        self.x_txt = tu.make_tensor_np(self.x_txt, config['w2i'], config['input_text_shape']) 

        # Create image features
        self.start_gif = time.time()
        self.x_img = np.squeeze(self.generate_image_features(self.img))
        self.end_gif = time.time()        

        self.start_pred = time.time()
        # Make model prediction
        with self.graph.as_default():
            self.scores = self.model.predict([self.x_img.reshape(-1, config['input_img_shape']), self.x_txt.reshape(-1, config['input_text_shape'])]).tolist()
            self.end_pred = time.time()
            

        self.classes = ['Aadhar Card', 'Diagnostic Bill', 'Discharge Summary', 'Insurance Card', 'Internal Case Papers', 'Pan Card', 'Phramacy Bill', 'Policy Copy', 'Prescriptions' , 'Receipts']
    
        # Prepare response and return
        self.scores_dict = dict(zip(self.classes, self.scores[0]))
        self.prediction = max(self.scores_dict.items(), key=operator.itemgetter(1))[0]
        print('Prediction:', self.prediction)
        
        self.response = dict({'prediction':self.prediction, 'imageid':self.imageid, 'scores_dict':self.scores_dict})
        self.end = time.time()
        
        self.dur = self.end - self.start
        
        self.dur_json = self.end_json - self.start_json
        self.dur_clean_string = self.end_clean_string - self.start_clean_string
        self.dur_gif = self.end_gif - self.start_gif
        self.dur_pred = self.end_pred - self.start_pred
        print("Execution Time decode:", self.dur_json,"seconds")
        print("Execution Time OCR:", self.dur_clean_string,"seconds")
        print("Execution Time img features:", self.dur_gif,"seconds")
        print("Execution Time pred:", self.dur_pred,"seconds")
    
        if self.dur<60:
            print("Execution Time:", self.dur,"seconds")
        elif self.dur>60 and self.dur<3600:
            self.dur=dur/60
            print("Execution Time:", self.dur,"minutes")
        else:
            self.dur = self.dur/(60*60)
            print("Execution Time:", self.dur,"hours")
        
        return json.dumps(self.response)
    
# Load the model 
model_path = "models/model2.h5"
loaded_model = LoadModel(model_path)
   
# App trigger
@app.route("/", methods=['GET', 'POST'])
def wrapper():
    req = request.get_data()   
    res = loaded_model.predict(req)
    return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import base64
import commands
import logging
import os

from tf_detector import TFDetector
from tf_recognizer import TFRecognizer
import cmt_dict
image_decoder = {
    '.jpg': tf.image.decode_jpeg,
    '.jpeg': tf.image.decode_jpeg,
    '.png': tf.image.decode_png,
    '.gif': tf.image.decode_gif
}

logging.getLogger("tensorflow").setLevel(logging.DEBUG)


class Predictor:

    def __init__(self, vehicle_graph_path, license_graph_path, ocr_graph_path, cmt_graph_path):
        self._vehicle_model = TFDetector(vehicle_graph_path)
        self._license_model = TFDetector(license_graph_path)
        self._ocr_model = TFRecognizer(ocr_graph_path, input_shape = [36,224], output_names=["prediction:0"])
        self._cmt_model = TFRecognizer(cmt_graph_path, 
            input_shape = [212,212], 
            output_names=["prediction_color:0", "prediction_make:0", "prediction_type:0"])
    def predict(self, image):
        result=[]
        #vihicle predict
        vehicle_box = self._vehicle_model.detect(image, 0.3)
        #license predict
        for i in range(len(vehicle_box)):
            rst = {}
            box = vehicle_box[i]
            rst['vehicle'] = box
            #cmt predict
            image_vehicle = image[box[1]:box[3],box[0]:box[2]]
            cmt = self._cmt_model.recognize(image_vehicle)
            c = cmt_dict.vehicle_color[cmt[0][0]]
            m = cmt_dict.vehicle_make[cmt[1][0]]
            t = cmt_dict.vehicle_type[cmt[2][0]]
            rst['color'] = c
            rst['make'] = m
            rst['type'] = t
            #license predict
            license_box = self._license_model.detect(image_vehicle, 0.2)
            if len(license_box)>0:
                license_box = license_box[0]
                rst['license'] = license_box
                #ocr predict
                image_license = image_vehicle[license_box[1]:license_box[3],license_box[0]:license_box[2]]
                ocr = self._ocr_model.recognize(image_license)
                rst['ocr'] = ocr[0][0]
            result.append(rst)
        return result
    def predict_bytes(self, name, data_bytes, height, width, channel):
        #decode image
        image = self.decode_image(name, data_bytes, height, width, channel)
        #predict
        return self.predict(image)
    def predict_image(self, image):
        return self.predict(image)
    def decode_image(self, name, data_bytes, height, width, channel):
        tf.logging.info('within predicting image')
        tf.logging.info(name)
        _, postfix = os.path.splitext(name)
        tf.logging.info(postfix)
        r = base64.decodestring(data_bytes)
        image = np.frombuffer(r, dtype=np.uint8)
        image = image.reshape((height, width, channel)) 
        return image

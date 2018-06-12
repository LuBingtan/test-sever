"""TF predictor

A predictor class to predict over a pre-trained tf graph. This class
bootstraps a TF session and load a pre-trained graph in init().
Data then can be fed in and do prediction using predict().

NOTE: pre-trained tf graph should be a binary protocal buffer file
having fixed value on weights/biases or other parameters in the graph.
See 'https://www.tensorflow.org/api_docs/python/tf/train/write_graph'
to write a binary graph, 'https://github.com/tensorflow/tensorflow/
blob/a0d784bdd31b27e013a7eac58a86ba62e86db299/tensorflow/python/tools
/freeze_graph.py' to freeze a graph with constant values.

NOTE: init() is not __init()__, which should be loaded manaully.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import base64
import commands
import logging
import os

from tf_detector import TFDetector
from tf_recognizer import TFRecognizer


logging.getLogger("tensorflow").setLevel(logging.DEBUG)


class Predictor:

    def init(self, vehicle_graph_path, license_graph_path, ocr_graph_path, cmt_graph_path):
        """Initialization of the class, including open a tf session and
        load a pre-trained tf graph. See class comments on top of this
        file for more information;

        Args:
            :params graph_path: a full path directing to a pre-trained
            binary .pb file.
        """
        '''
        # Import the graph.
        with tf.gfile.FastGFile(self.graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name = '')
            tf.logging.info("Model restored.")

        ## 'input' and 'prediction' op should be named when storing
        ## the graph.
        self.input = self.sess.graph.get_tensor_by_name('features:0')
        self.output = self.sess.graph.get_tensor_by_name('prediction:0')
        '''
        self.vehicle_model = TFDetector(vehicle_graph_path)
        self.license_model = TFDetector(license_graph_path)
        self.ocr_model = TFRecognizer(ocr_graph_path)
        self.cmt_model = TFRecognizer(cmt_graph_path)

    def predict(self, instances):
        """ This method predict fed-in instance using pre-trained model.
        This function should be called after manually `init()` this class.
    
        Args:
            instances: a 2-D list of data input.
        Returns:
            A 2-D ndarray of prediction result.
            
        """
        return self.sess.run(self.output, feed_dict = {self.input: instances})
        

    def predict_image(self, name, bytes, height, width, channel):
        tf.logging.info('within predicting image')
        tf.logging.info(name)

        _, postfix = os.path.splitext(name)
        tf.logging.info(postfix)

        decoded_bytes = base64.b64decode(bytes)
        decoder = image_decoder[postfix]
        image_matrix = decoder(decoded_bytes)
        resized_image = tf.image.resize_images(image_matrix, [height, width])
        resized_image.set_shape((height, width, channel))

        decoded_image = self.sess.run(resized_image)
        return self.sess.run(self.output, feed_dict = {self.input: [decoded_image]})

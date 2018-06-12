from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from recognizer import Recognizer

class TFRecognizer(Recognizer):

    def __init__(self, graph_path, input_shape=[36,224], output_names=['prediction:0'], legacy=False):
        self._input_shape = input_shape
        self._legacy = legacy
        self._output_names = output_names
        self._graph = self._init_graph(graph_path)
        self._sess = tf.Session(graph=self._graph)
        #self._sess.run([tf.local_variables_initializer(), tf.tables_initializer()])

        self._letters_lower = list('abcdefghijklmnopqrstuvwxyz')
        self._letters_upper = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self._numbers = list('0123456789')
        self._output_size = (31, 26, 36, 36, 36, 36, 36)


    def clear(self):
        """Clear anything everything."""
        tf.reset_default_graph()
        self._sess.close()


    def recognize(self, image):
        """Recognize based on plate image.
        Args:
            plate_image: plate image to be recognized
        Returns:
            should return recognized characters
        """
        a = self.recognize_batch([image])
        return self.recognize_batch([image])[0]


    def recognize_batch(self, images):
        """Recognize based on plate images.
        Args:
            plate_images: an array of plate_images to be recognized
        Returns:
            should return an array of recognized characters
        """
        preds = []
        try:
            if self._legacy:
                pred = self._recognize_legacy(images)
                preds.extend(pred)
            else:
                resized_images = []
                for image in images:
                    resized_image = np.resize(image, (self._input_shape[0], self._input_shape[1], 1))
                    resized_images.append(resized_image)
                pred = self._sess.run(self._prediction, feed_dict={self._features: resized_images})
                preds.append(pred)
        finally:
            return preds


    def _recognize_legacy(self, images):
        preds = self._sess.run(self._prediction, feed_dict={self._features: images})
        return [self._pred2word(pred) for pred in preds]


    def _init_graph(self, graph_path):
        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')
            if self._legacy:
                self._features = graph.get_tensor_by_name('x:0')
                self._prediction = graph.get_tensor_by_name('softmax:0')
            else:
                self._features = graph.get_tensor_by_name('features:0')
                self._prediction = []
                for name in self._output_names:
                    self._prediction.append(graph.get_tensor_by_name(name))
        tf.logging.info("Model restored.")
        return graph


    def _pred2word(self, pred):
        prev = 0
        word = ''
        for i in range(len(self._output_size)):
            cur = pred[prev: prev + self._output_size[i]]
            index = np.argmax(cur)
    
            dictionary = self._letters_upper + self._numbers
            if i == 0:
                dictionary = self._letters_lower + self._numbers
            if i == 1:
                dictionary = self._letters_upper
    
            prev += self._output_size[i]
            word += dictionary[index]
    
        return word

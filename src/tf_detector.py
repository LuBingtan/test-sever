"""TF detector is in charge of localizing objects in the image(s),
either a vehicle or a vehicle license plate.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from detector import Detector


class TFDetector(Detector):

    def __init__(self, graph_path):
        self._graph = self._init_graph(graph_path)
        #self._category_index = self._init_category_index(label_map)

        self._sess = tf.Session(graph=self._graph)

    def clear(self):
        """Clear anything everything."""
        self._sess.close()

    def detect(self, image, threshold=0.2):
        """Detect desired objects in the image.
        Args:
            image: image to be detected
            threshold: threshold to filter images
        Returns:
            should return detected boundary boxes
        """
        height = image.shape[0]
        width = image.shape[1]
        image_expanded = np.expand_dims(image, axis=0)

        boxes, scores, classes, num_detections = self._sess.run(
                [self._boxes, self._scores, self._classes, self._num_detections],
                feed_dict={self._image_tensor: image_expanded})

        todo_scores = np.squeeze(scores)
        todo_boxes = [
            boxes[0][i] for i in range(len(todo_scores))
            if todo_scores[i] > threshold]
        scaled_boxes = [
            (int(box[1] * width),
             int(box[0] * height),
             int(box[3] * width),
             int(box[2] * height)) for box in todo_boxes]
        return scaled_boxes

    def _init_graph(self, graph_path):
        """Initialize tensorflow graph.
        This function collects tensors from the pre-trained model.

        Args:
            graph_path: a tensorflow graph file
        """
        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

            self._image_tensor = graph.get_tensor_by_name('image_tensor:0')
            self._boxes = graph.get_tensor_by_name('detection_boxes:0')
            self._scores = graph.get_tensor_by_name('detection_scores:0')
            self._classes = graph.get_tensor_by_name('detection_classes:0')
            self._num_detections = graph.get_tensor_by_name('num_detections:0')

        return graph
    '''
    def _init_category_index(self, label_map):
        """Initialize category index.

        Args:
            label_map: a tensorflow model label_map file
        """
        label_map = label_map_util.load_labelmap(label_map)
        categories = label_map_util.convert_label_map_to_categories(
                label_map, max_num_classes=1, use_display_name=True)

        self._category_index = label_map_util.create_category_index(categories)
    '''

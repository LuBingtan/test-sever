#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Dieu web server.

A simple http server listening to tf prediction request and
responding with prediction result.
"""
import numpy as np
import time
import json
import logging
import tensorflow as tf
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

from predictor import Predictor
vehicle_graph_path = "../models/frozen_inference_graph.pb"
license_graph_path = "../models/faster_rcnn_plate_graph.pb"
ocr_graph_path = "../models/ocr_1029_753000s.model"
cmt_graph_path = "../models/simple_frozen_inference_graph_1_3.pb"
predictor = Predictor(vehicle_graph_path, license_graph_path, ocr_graph_path, cmt_graph_path)
logging.getLogger("tensorflow").setLevel(logging.DEBUG)
'''test 
import cv2 as cv
image = cv.imread("test.jpg")
rst = predictor.predict_image(image)
print "--------------------------rst--------------------------"
print rst
print "--------------------------rst--------------------------"
print json.dumps(rst)
'''
class DieuHTTPHandler(BaseHTTPRequestHandler):

    def _set_headers(self):
        """Set headers. """
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()


    def do_POST(self):
        """Predict when receiving POST request. """
        self._set_headers()
        tf.logging.info('Receive POST request, begin to predict!\n')

        data_string = self.rfile.read(int(self.headers['Content-Length']))
        tf.logging.info("data get")

        # try:
        # Load the json data and do prediction here.
        data_json = json.loads(data_string)

        if "data" in data_json:
            print(type(data_json['data']))
            result = predictor.predict_image(np.array(data_json['data']))
        else:
            name = data_json['name']
            bytes = data_json['bytes']
            height = data_json['height']
            width = data_json['width']
            channel = data_json['channel']
            result = predictor.predict_bytes(name, bytes, height, width, channel)
            for t in result:
                tf.logging.debug(t)


        result_json = json.dumps(result)
        self.wfile.write(result_json)
        # except Exception as e:
        #     self.wfile.write(e)
        #     raise e


def run(server = HTTPServer, handler = DieuHTTPHandler,
        port = 7777):
    """Start a Dieu web service. Default starting port is 7777.

    Args:
        :param graph_path: the place storing the graph which
        is a protocal buffer file.
        :param server: a HTTP Server;
        :param handler: a handler dealing with HTTP request;
        :param port: the port to start the server;
    """
    #predictor.init(graph_path)
    address = ('', port)
    httpd = server(address, handler)
    tf.logging.info('Dieu web server launched!')
    httpd.serve_forever()


if __name__ == '__main__':
    from sys import argv

    # Use the port if it is specified.
    '''
    if len(argv) > 2:
        run(argv[1], port = int(argv[2]))
    else:
        run(argv[1])
    '''
    run()
#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Dieu web server.

A simple http server listening to tf prediction request and
responding with prediction result.
"""

import json
import logging
import tensorflow as tf
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

#from predictor import Predictor


#predictor = Predictor()
logging.getLogger("tensorflow").setLevel(logging.DEBUG)


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
        tf.logging.info(data_string)

        # try:
        # Load the json data and do prediction here.
        data_json = json.loads(data_string)

        if "data" in data_json:
            print("data")
            #result = predictor.predict(data_json['data'])
        else:
            name = data_json['name']
            bytes = data_json['bytes']
            height = data_json['height']
            width = data_json['width']
            channel = data_json['channel']
            #result = predictor.predict_image(name, bytes, height, width, channel)

        result_json = json.dumps(result.tolist())
        self.wfile.write(result_json)
        # except Exception as e:
        #     self.wfile.write(e)
        #     raise e


def run(graph_path, server = HTTPServer, handler = DieuHTTPHandler,
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
    if len(argv) > 2:
        run(argv[1], port = int(argv[2]))
    else:
        run(argv[1])

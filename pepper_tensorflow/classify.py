# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

## ..:: Adapted from the TensorFlow Object Recognition Tutorial ::.. ##
## https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py


import tensorflow as tf
import numpy as np

from threading import Thread
from socketserver import TCPServer, BaseRequestHandler
from socket import socket
import yaml

import os
import re
from time import time, strftime

MODEL_DIR = os.path.join(os.path.dirname(__file__), r"model/inception")


class NodeLookup:
    LABEL_LUT = os.path.join(MODEL_DIR, r'imagenet_2012_challenge_label_map_proto.pbtxt')
    HUMAN_LUT = os.path.join(MODEL_DIR, r'imagenet_synset_to_human_label_map.txt')

    def __init__(self):
        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(self.HUMAN_LUT).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(self.LABEL_LUT).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        self.node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            self.node_id_to_name[key] = name

    def get(self, node_id: str) -> str:
        return self.node_id_to_name[node_id]


class Classify:
    IMAGE_GRAPH = os.path.join(MODEL_DIR, r'classify_image_graph_def.pb')

    def __init__(self, n_predictions: int = 1):
        """
        Classify Jpeg images

        Parameters
        ----------
        n_predictions: int
            Number of predictions to make per image
        """

        with tf.gfile.FastGFile(self.IMAGE_GRAPH, 'rb') as graph_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(graph_file.read())
            tf.import_graph_def(graph_def, name='')

        self.node = NodeLookup()
        self.n_predictions = n_predictions
        self.session = tf.Session()
        self.softmax = self.session.graph.get_tensor_by_name('softmax:0')

    def classify(self, jpeg: bytes):
        """
        Classify Jpeg

        Parameters
        ----------
        jpeg: bytes
            Jpeg image to classify

        Returns
        -------
        classification: list of (float, list)
            List of confidence-object pairs
        """
        with tf.device('/cpu:0'):
            predictions = np.squeeze(self.session.run(self.softmax, {'DecodeJpeg/contents:0': jpeg}))

        top_predictions = predictions.argsort()[-self.n_predictions:][::-1]

        return [[float(predictions[node_id]), self.node.get(node_id).split(', ')] for node_id in top_predictions]

    def __del__(self):
        self.session.close()


class ClassifyRequestHandler(BaseRequestHandler):

    CLASSIFY = Classify()

    def handle(self):
        """
        Handle client classification request
        """
        try:

            t0 = time()

            # Receive Jpeg
            jpeg_size = int(np.frombuffer(self.request.recv(4), np.uint32)[0])  # Jpeg size is transmitted as uint32
            jpeg = self._receive_bytes(jpeg_size)  # Jpeg is transmitted as 'jpeg_size' bytes

            # Classify Jpeg
            classification = self.CLASSIFY.classify(jpeg)

            # Return yaml-encoded classification
            self.request.sendall(yaml.dump(classification).encode())

            # Print statistics
            print("[{}][{:3.2f}s] {}: [{:3.0%}] {}".format(
                strftime("%H:%M:%S"), time() - t0, self.client_address,
                classification[0][0], classification[0][1]))

        # If the client disconnects, that is ok
        except ConnectionResetError:
            pass

    def _receive_bytes(self, n: int):
        """
        Receive exactly n bytes

        Parameters
        ----------
        n: int
            Number of bytes to receive

        Returns
        -------
        bytes
        """

        jpeg_buffer = bytearray()

        while len(jpeg_buffer) < n:
            jpeg_buffer.extend(self.request.recv(4096))

        return bytes(jpeg_buffer)


class ClassifyServer(TCPServer):
    def __init__(self, port: int, deamon: bool = False):
        """
        Run Classify Server

        Parameters
        ----------
        port: int
            Port to listen to for incoming classification requests
        deamon: bool
            Whether server thread should be daemon or not
        """
        super().__init__(('', port), ClassifyRequestHandler)
        Thread(target=self.serve_forever, daemon=deamon).start()


class ClassifyClient:
    def __init__(self, address: tuple):
        """
        Sample Classification Client

        Parameters
        ----------
        address: (str, int)
            Address of Classification Host
        """

        self.address = address

    def classify(self, path: str):
        """
        Classify Jpeg Image on disk

        Parameters
        ----------
        path: str
            Path to Jpeg image on disk

        Returns
        -------
        classification: list of (float, list)
            List of confidence-object pairs
        """
        sock = socket()

        sock.connect(self.address)

        with open(path, 'rb') as jpeg_file:
            jpeg = jpeg_file.read()
            jpeg_size = np.array([len(jpeg)], np.uint32)

        sock.sendall(jpeg_size)
        sock.sendall(jpeg)

        response = yaml.load(sock.recv(4096).decode())

        sock.close()

        return response


if __name__ == "__main__":
    server = ClassifyServer(9999)
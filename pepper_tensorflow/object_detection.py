import tensorflow as tf
import numpy as np

from threading import Thread
from socketserver import TCPServer, BaseRequestHandler
from socket import socket

from time import time

import json
import re
import os

from typing import Dict, Iterable


class ObjectDetectionModelPath:
    def __init__(self, name, root, graph, labels):
        self.name = name
        self.root = root
        self.graph = os.path.join(self.root, graph)
        self.labels = os.path.join(self.root, labels)


class ObjectDetectionModel:
    AVA = ObjectDetectionModelPath(
        name="AVA",
        root=os.path.join(os.path.dirname(__file__), 'model/ava'),
        graph='faster_rcnn_resnet101_ava_v2.1_2018_04_30/frozen_inference_graph.pb',
        labels='labelmap.json')
    COCO = ObjectDetectionModelPath(
        name="COCO",
        root=os.path.join(os.path.dirname(__file__), 'model/coco'),
        graph='ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb',
        labels='labelmap.json')
    OID = ObjectDetectionModelPath(
        name="OID",
        root=os.path.join(os.path.dirname(__file__), 'model/oid'),
        graph='faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/frozen_inference_graph.pb',
        labels='labelmap.json')


class Object:
    def __init__(self, name, score, box):
        self._name = name
        self._score = score
        self._box = box

    @property
    def name(self) -> str:
        return self._name

    @property
    def score(self) -> float:
        return self._score

    @property
    def box(self):
        return self._box

    @classmethod
    def from_dict(cls, dictionary):
        return cls(dictionary['name'], dictionary['score'], dictionary['box'])

    def to_dict(self) -> Dict:
        return {'name': self.name, 'score': self.score, 'box': self.box}

    def __str__(self):
        return f"[{self.score:4.0%}] {self.name:10s} {self.box}"


class ObjectDetection:

    CLASSES = 'detection_classes:0'
    SCORES = 'detection_scores:0'
    BOXES = 'detection_boxes:0'

    def __init__(self, model: ObjectDetectionModelPath):

        t0 = time()

        self._model = model

        with open(self.model.labels) as label_file:
            self._labels = {int(key): value for key, value in json.load(label_file).items()}

        self._graph = tf.Graph()

        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model.graph, 'rb') as graph_file:
                serialized_graph = graph_file.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

            self._session = tf.Session(graph=self.graph)

            self._tensor_dict = {name: self.graph.get_tensor_by_name(name) for name in
                                 [self.BOXES, self.SCORES, self.CLASSES]}
            self._image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        print(f"[{time() - t0:4.1f}s] {self.__str__()} Booted")

    @property
    def model(self) -> ObjectDetectionModelPath:
        return self._model

    @property
    def labels(self) -> Dict[int, Dict]:
        return self._labels

    @property
    def graph(self) -> tf.Graph:
        return self._graph

    @property
    def session(self):
        return self._session

    def classify(self, image: np.ndarray) -> Iterable[Object]:
        output = self.session.run(self._tensor_dict, {self._image_tensor: np.expand_dims(image, 0)})
        for index, score, box in zip(output[self.CLASSES][0], output[self.SCORES][0], output[self.BOXES][0]):
            yield Object(str(self._labels[int(index)]['name']), float(score), box.tolist())

    @staticmethod
    def pbtxt_to_json(labels_pbtxt):
        from collections import OrderedDict
        dictionary = OrderedDict()

        with open(labels_pbtxt) as label_file:
            for item in re.findall('item {(.+?)}', label_file.read().replace("\n", " ")):

                index = int(re.findall("id: (\d+)", item)[0])
                name = re.findall("display_name: \"(.+?)\"", item)[0]

                dictionary[index] = {'name': name, 'id': index}

        path, ext = os.path.splitext(labels_pbtxt)
        with open(path + ".json", 'w') as json_file:
            json.dump(dictionary, json_file, indent=0)

    def __str__(self):
        return f"{self.__class__.__name__}({self.model.name})"


class ObjectDetectionRequestHandler(BaseRequestHandler):
    def handle(self):
        try:
            t0 = time()

            width, height, channels = np.frombuffer(self.request.recv(3*4), np.uint32)
            image = self._receive_image(width, height, channels)
            response = json.dumps([obj.to_dict() for obj in self.server.classifier.classify(image)])

            print(f"[{time() - t0:3.2f}s] {self.server.classifier.__str__():20s} {response}")

            self.request.sendall(np.uint32(len(response)))
            self.request.sendall(response.encode())

        except ConnectionResetError:
            pass

    def _receive_image(self, width: int, height: int, channels: int) -> np.ndarray:
        buffer = bytearray()
        n = width * height * channels
        while len(buffer) < n:
            buffer.extend(self.request.recv(4096))
        return np.frombuffer(buffer, np.uint8).reshape(width, height, channels)


class ObjectDetectionServer(TCPServer):
    def __init__(self, classifier: ObjectDetection, port: int, daemon: bool = False):
        super().__init__(('', port), ObjectDetectionRequestHandler)
        self._classifier = classifier
        Thread(target=self.serve_forever, daemon=daemon).start()

    @property
    def classifier(self):
        return self._classifier


class ObjectDetectionClient:
    def __init__(self, address: tuple):
        self._address = address

    def classify(self, image: np.ndarray):
        sock = socket()
        sock.connect(self._address)

        sock.sendall(np.array(image.shape, np.uint32).tobytes())
        sock.sendall(image.tobytes())

        response_length = np.frombuffer(sock.recv(4), np.uint32)[0]
        response = [Object.from_dict(info) for info in json.loads(self._receive_all(sock, response_length).decode())]

        return response

    @staticmethod
    def _receive_all(sock: socket, n: int) -> bytearray:
        buffer = bytearray()
        while len(buffer) < n:
            buffer.extend(sock.recv(4096))
        return buffer


if __name__ == '__main__':
    # # Translate .pbtxt to .json
    # for src in [ObjectDetectionModel.COCO, ObjectDetectionModel.OID, ObjectDetectionModel.AVA]:
    #     path, ext = os.path.splitext(src.labels)
    #     ObjectDetection.pbtxt_to_json(path + '.pbtxt')

    AVA_port, COCO_port, OID_port = 27001, 27002, 27003

    # AVA_server = ObjectDetectionServer(ObjectDetection(ObjectDetectionModel.AVA), AVA_port)
    COCO_server = ObjectDetectionServer(ObjectDetection(ObjectDetectionModel.COCO), COCO_port)
    # OID_server = ObjectDetectionServer(ObjectDetection(ObjectDetectionModel.OID), OID_port)

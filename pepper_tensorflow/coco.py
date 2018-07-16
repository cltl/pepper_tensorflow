from pepper_tensorflow.model.coco import label_map_util, visualization_utils
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import json

from threading import Thread
from socketserver import TCPServer, BaseRequestHandler
from socket import socket

from time import time
import os


class Classify:

    PATH_TO_GRAPH = os.path.abspath('model/coco/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.abspath('model/coco/mscoco_label_map.pbtxt')
    NUM_CLASSES = 90

    def __init__(self):
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.session = tf.Session(graph=self.graph)

            # Get handles to input and output tensors
            ops = self.graph.get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = self.graph.get_tensor_by_name(tensor_name)
            
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

    def classify(self, image):
        output_dict = self.session.run(self.tensor_dict, feed_dict={self.image_tensor: np.expand_dims(image, 0)})
        return [self.category_index[int(cls)] for cls in output_dict['detection_classes'][0]],\
               output_dict['detection_scores'][0].tolist(),\
               output_dict['detection_boxes'][0].tolist()

    def visualize(self, image):
        classes, scores, boxes = self.classify(image)
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image, boxes, [cls['id'] for cls in classes], scores, self.category_index, use_normalized_coordinates=True)
        plt.imshow(image)
        plt.show()

    @staticmethod
    def _reframe_box_masks_to_image_masks(box_masks, boxes, image_height, image_width):
        """Transforms the box masks back to full image masks.

        Embeds masks in bounding boxes of larger masks whose shapes correspond to
        image shape.

        Args:
          box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
          boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
                 corners. Row i contains [ymin, xmin, ymax, xmax] of the box
                 corresponding to mask i. Note that the box corners are in
                 normalized coordinates.
          image_height: Image height. The output mask will have the same height as
                        the image height.
          image_width: Image width. The output mask will have the same width as the
                       image width.

        Returns:
          A tf.float32 tensor of size [num_masks, image_height, image_width].
        """
        def transform_boxes_relative_to_boxes(boxes, reference_boxes):
            boxes = tf.reshape(boxes, [-1, 2, 2])
            min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
            max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
            transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
            return tf.reshape(transformed_boxes, [-1, 4])

        box_masks = tf.expand_dims(box_masks, axis=3)
        num_boxes = tf.shape(box_masks)[0]
        unit_boxes = tf.concat(
            [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
        reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
        image_masks = tf.image.crop_and_resize(image=box_masks,
                                               boxes=reverse_boxes,
                                               box_ind=tf.range(num_boxes),
                                               crop_size=[image_height, image_width],
                                               extrapolation_value=0.0)
        return tf.squeeze(image_masks, axis=3)


class ClassifyRequestHandler(BaseRequestHandler):

    CLASSIFY = Classify()

    def handle(self):
        try:
            width, height, channels = np.frombuffer(self.request.recv(3*4), np.uint32)
            image = self._receive_image(width, height, channels)
            response = json.dumps(self.CLASSIFY.classify(image))

            print("Image({}, {}, {}) -> {}".format(width, height, channels, response))

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


class ClassifyServer(TCPServer):

    PORT = 35621

    def __init__(self, port: int = PORT, daemon: bool=False):
        super().__init__(('', port), ClassifyRequestHandler)
        Thread(target=self.serve_forever, daemon=daemon).start()


class ClassifyClient:

    def __init__(self, address: tuple = ('localhost', ClassifyServer.PORT)):
        self.address = address

    def classify(self, image: np.ndarray):
        sock = socket()
        sock.connect(self.address)

        sock.sendall(np.array(image.shape, np.uint32))
        sock.sendall(image)

        response_length = np.frombuffer(sock.recv(4), np.uint32)[0]
        response = json.loads(self._recv_all(sock, response_length).decode())

        return response

    def _recv_all(self, sock: socket, n: int) -> bytearray:
        buffer = bytearray()
        while len(buffer) < n:
            buffer.extend(sock.recv(4096))
        return buffer


if __name__ == "__main__":
    ClassifyServer()
    print("Running!")

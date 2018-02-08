import tensorflow as tf
import numpy as np

from socketserver import TCPServer, BaseRequestHandler
from threading import Thread
import os


class GenderModel:

    MODEL_PATH = os.path.join("model", "gender", "gender.ckpt")

    def __init__(self):
        """Model for Gender Estimation from 128D OpenFace Face Vectors"""

        ## Model Description ##
        self.x = tf.placeholder(tf.float32, [None, 128])            # Placeholder for 128 Dimensional Face Vectors
        self.W = tf.Variable(tf.zeros([128, 2]))                    # Weights
        self.b = tf.Variable(tf.zeros([2]))                         # Biases
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)  # Softmax Function
        self.y_ = tf.placeholder(tf.float32, [None, 2])             # Correct Answers

    def train(self, face: np.ndarray, gender: np.ndarray, learning_rate = 0.5, training_steps = 2000):
        """
        Train Model with Training Data, Saving Results to File

        Parameters
        ----------
        face: np.ndarray
            Face Array in shape (N, 128)
        gender: np.ndarray
            Gender Array in shape (N, 2) - One Hot Encoded
        learning_rate: float
            Model Learning Rate
        training_steps: int
            Model Training Steps
        """
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print("Train Model, with {} data entries".format(len(face)))

            for i in range(training_steps):
                train_step.run(feed_dict={self.x: face, self.y_: gender})
                print("\rTraining Step {}/{}".format(i + 1, training_steps), end='')
            tf.train.Saver().save(session, self.MODEL_PATH)
            print("\rSaved model in {}".format(os.path.abspath(self.MODEL_PATH)))

    def test(self, face: np.ndarray, gender: np.ndarray):
        """
        Test Model with Test Data, Printing Accuracy

        Parameters
        ----------
        face: np.ndarray
            Face Array in shape (N, 128)
        gender: np.ndarray
            Gender Array in shape (N, 2) - One Hot Encoded
        """
        with tf.Session() as session:
            tf.train.Saver().restore(session, self.MODEL_PATH)
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_evaluation = accuracy.eval(feed_dict={self.x: face, self.y_: gender})
            print("Accuracy: {}".format(accuracy_evaluation))

    @staticmethod
    def load_data(target: str = 'wiki') -> (np.ndarray, np.ndarray):
        """
        Conveniently Load Data from Disk

        Parameters
        ----------
        target: str
            either 'wiki' or 'imdb'

        Returns
        -------
        data: (np.ndarray, np.ndarray)
            (face, gender) tuple
        """
        FACE_PATH = "data/gender/{}/matrix.bin".format(target)
        GENDER_PATH = "data/gender/{}/gender_age.bin".format(target)
        face = np.fromfile(FACE_PATH, np.float32).reshape(-1, 128)
        gender_age = np.fromfile(GENDER_PATH, np.byte).view([('gender',np.bool), ('age',np.float32)]).view(np.recarray)
        gender = np.eye(2)[gender_age['gender'].astype(np.int32)]  # One Hot Encoding of Gender
        return face, gender


class GenderClassify(GenderModel):
    def __init__(self):
        """Gender Classification, using Saved Model from Disk"""

        super().__init__()

        self.session = tf.Session()
        tf.train.Saver().restore(self.session, self.MODEL_PATH)

    def classify(self, face: np.ndarray) -> np.ndarray:
        """
        Classify Face(s)

        Parameters
        ----------
        face: np.ndarray
            Face(s) to classify, encoded as 128D vectors, according to OpenFace spec

        Returns
        -------
        evaluation: np.ndarray
            Per Face P("Female") == 1-P("Male")
        """
        return self.y.eval(feed_dict={self.x: face}, session=self.session)[:, 0]


class ClassifyRequestHandler(BaseRequestHandler):
    """Handle Client Gender Classification Requests"""

    CLASSIFY = GenderClassify()
    BUFFER_SIZE = 4096

    def handle(self):
        """Handle Classification Request"""

        print("Got Request")

        # Receive int32 indicating how many bytes will be sent
        n_bytes = np.frombuffer(self.request.recv(4), np.int32)

        # Receive n_bytes bytes with face(s)
        face_buffer = bytearray()
        while len(face_buffer) < n_bytes:
            face_buffer.extend(self.request.recv(self.BUFFER_SIZE))
        face = np.frombuffer(face_buffer, np.float32).reshape(-1, 128)

        classification = self.CLASSIFY.classify(face)

        # Send Number of Bytes to be sent Back
        self.request.sendall(np.int32(len(classification)))

        # Send all classification result bytes
        self.request.sendall(classification.tobytes())


class ClassifyServer(TCPServer):
    def __init__(self, port: int = 8678, daemon: bool = False):
        """
        Run Gender Classify Server

        See pepper/vision/face.py/GenderClassifyClient for corresponding Client program

        Parameters
        ----------
        port: int
            Port to listen to for incoming classification requests
        deamon: bool
            Whether server thread should be daemon or not
        """
        super().__init__(('', port), ClassifyRequestHandler)
        Thread(target=self.serve_forever, daemon=daemon).start()
        print("Gender Classification Server Booted")


if __name__ == "__main__":
    ClassifyServer()
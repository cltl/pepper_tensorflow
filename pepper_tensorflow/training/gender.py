import tensorflow as tf
import numpy as np
import os


MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "model", "gender", "gender.ckpt"))
FACE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data", "wiki", "matrix.bin"))
GENDER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data", "wiki", "gender_age.bin"))

face = np.fromfile(FACE_PATH, np.float32).reshape(-1, 128)
gender_age = np.fromfile(GENDER_PATH, np.byte).view([('gender', np.bool), ('age', np.float32)]).view(np.recarray)

gender = np.eye(2)[gender_age['gender'].astype(np.int32)]  # One Hot Encoding of Gender

# Split Data into Train and Test Data
train_size = len(face) // 2

split = np.zeros(len(face), np.bool)
split[np.random.choice(np.arange(len(face)), train_size)] = True

train = face[split], gender[split]
test = face[~split], gender[~split]

## Training Parameters ##
LEARNING_RATE = 0.5
TRAINING_STEPS = 2000

## Model Description ##

# Placeholder for 128D Face Vectors
x = tf.placeholder(tf.float32, [None, 128])

# 'Model'
W = tf.Variable(tf.zeros([128, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Placeholder for Correct Answers
y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    print("Run Model")
    print("Train Data: {}, Test Data: {}".format(len(train[0]), len(test[0])))
    print("Learning Rate: {}".format(LEARNING_RATE))


    for i in range(TRAINING_STEPS):
        train_step.run(feed_dict={x: train[0], y_: train[1]})
        print("\rTraining Step {}/{}".format(i+1, TRAINING_STEPS), end='')

    print()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: {}".format(accuracy.eval(feed_dict={x: test[0], y_:test[1]})))

    tf.train.Saver().save(session, MODEL_PATH)
    print("Saved Model in path: {}".format(MODEL_PATH))


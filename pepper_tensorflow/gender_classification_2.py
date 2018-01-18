import tensorflow as tf
import numpy as np

np.random.seed(0)

# Load Data
FACE_PATH = r"data/wiki/wiki_matrix.bin"
GENDER_PATH = r"data/wiki/wiki_gender_age.bin"

face = np.fromfile(FACE_PATH, np.float32).reshape(-1, 128)
gender_age = np.fromfile(GENDER_PATH, np.byte).view([('gender', np.bool), ('age', np.float32)]).view(np.recarray)

gender = np.eye(2)[gender_age['gender'].astype(np.int32)]  # One Hot Encoding of Gender


# Split Data into Train and Test Data
train_size = (len(face) // 5) * 4

split = np.zeros(len(face), np.bool)
split[np.random.choice(np.arange(len(face)), train_size)] = True

train = face[split], gender[split]
test = face[~split], gender[~split]


# Model Description

VECTOR_SIZE = 128
HIDDEN_UNITS_1 = 16
HIDDEN_UNITS_2 = 8
NUMBER_OF_CLASSES = 2

LEARNING_RATE = 0.1
TRAINING_STEPS = 10000


# Placeholder for 128D Face Vectors
x = tf.placeholder(tf.float32, [None, 128])

# 'Model'
with tf.name_scope('hidden_1'):
    W = tf.Variable(tf.truncated_normal([VECTOR_SIZE, HIDDEN_UNITS_1], stddev=1.0))
    b = tf.Variable(tf.zeros([HIDDEN_UNITS_1]))
    h1 = tf.nn.sigmoid(tf.matmul(x, W) + b)
with tf.name_scope('hidden_2'):
    W = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_1, HIDDEN_UNITS_2], stddev=1.0))
    b = tf.Variable(tf.zeros([HIDDEN_UNITS_2]))
    h2 = tf.nn.sigmoid(tf.matmul(h1, W) + b)
with tf.name_scope('softmax_linear'):
    W = tf.Variable(tf.truncated_normal([HIDDEN_UNITS_2, NUMBER_OF_CLASSES], stddev=1.0))
    b = tf.Variable(tf.zeros([NUMBER_OF_CLASSES]))
    y = tf.matmul(h2, W) + b

# Placeholder for Correct Answers
y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


for i in range(TRAINING_STEPS):
    train_step.run(feed_dict={x: train[0], y_: train[1]})
    print("\rTraining Step {}/{}".format(i+1, TRAINING_STEPS), end='')

print()


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy: {}".format(accuracy.eval(feed_dict={x: test[0], y_:test[1]})))
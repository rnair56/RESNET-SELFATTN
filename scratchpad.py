import tensorflow as tf
import numpy as np
# import tensorflow.keras as k
from tensorflow.keras.layers import Dense, Conv2D

batch_size, n, m, k = 10, 3, 5, 2
A = tf.Variable(tf.random.normal(shape=(batch_size, n, m)))
B = tf.Variable(tf.random.normal(shape=(batch_size, m, k)))
# tf.matmul(A, B)
c = A @ B
print(c.shape)

# D = tf.Variable(tf.random.normal(shape=(None, 64)))
# print(D.shape)
# D = tf.reshape(D, (None, 1, 64))
# print(D.shape)
# print(type(D))


conv1 = Conv2D(filters=64, kernel_size=(2, 2))
den1 = Dense(64)
## input is a placeholder,
# input = tf.keras.Input(shape=(64, 64,3), batch_size=100)
# convout = conv1(input)
# denout = den1(convout)
#
# model = tf.keras.models.Model(input, denout)
# model.build(input_shape=(1000, 64, 64, 3))
#
# print(model.summary())
#
# model.compile(loss='mse')
# ones = tf.ones(shape=(100, 64,64,3))
#
# model.fit(ones, y=tf.zeros(shape=(100,)))
#
# a = np.array([1, 3, 5, 2])
# label = tf.keras.utils.to_categorical(a, 98)
# print(label.shape)

mat = np.array([[3, 2, 1],
           [2, 1, 3],
           [1, 3, 2]])
print("mat shape", mat.shape)
print(tf.sort(mat, axis=-1).numpy())
print("sort axis=0 \n", tf.sort(mat, axis=0).numpy())

A = np.array([[2, 2, 1, 0, 8],
           [8, 2, 0, 3, 7],
           [3, 2, 6, 5, 3],
           [1, 4, 2, 5, 8],
           [2, 3, 7, 0, 3]])

B = np.array([[3, 7, 6, 8, 3],
           [0, 7, 4, 4, 3],
           [1, 2, 0, 0, 4],
           [8, 6, 6, 7, 1],
           [8, 1, 0, 4, 8]])
mask = A.argsort()
print(mask)

test = np.argsort(B*-1)
print(test)

p = tf.constant([[3, 7, 6, 8, 3],
           [0, 7, 4, 4, 3],
           [1, 2, 0, 0, 4],
           [8, 6, 6, 7, 1],
           [8, 1, 0, 4, 8]])

print(np.argsort(p*-1))

#-*- coding:utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
from lle import LLE

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("F:\program\dataset\mnist", one_hot=True)

num = 10


X = mnist.train.images[0: num]
Y = mnist.train.labels[0: num]
uni_Y = np.unique(Y)

# print(X.shape)
# print(Y.shape)
# print(uni_Y)


for i in range(10):
    ax = plt.subplot(2, 10, i+1, frame_on=False)
    image = np.reshape(X[i, :], [28, -1])
    ax.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.title('Original # ' + str(i+1))
plt.show()

lle = LLE(k_neighbors=10, low_dims=400)
low_x = lle.fit_transform(X)
# print(low_x.shape)











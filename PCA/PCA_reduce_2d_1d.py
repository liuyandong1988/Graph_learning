#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2019/11/19 6:35
# PCA: 2D reduce to 1D

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import copy

con_matrix = np.array([[-0.9, 0.4], [0.1, 0.2]])
N = 300   # the node number

# # centralization
# original_data = (np.matlib.randn(N, 2) * con_matrix).A  # the original data
# x_mean = original_data.mean(axis=1)
# original_data_mean = copy.copy(original_data)
# original_data_mean[:, 0] = original_data[:, 0]-x_mean
# original_data_mean[:, 1] = original_data[:, 1]-x_mean

# no centralization
original_data = (np.matlib.randn(N, 2) * con_matrix).A  # the original data
original_data_mean = copy.copy(original_data)
covariance_data = 1/N * np.dot(original_data_mean.T, original_data_mean)  # covariance

# calculate the eigenvalue and eigenvector
eigenvalue, eigenvector = np.linalg.eig(covariance_data)
# eigenvalue sort
eigenvalue_index_sort = np.argsort(-eigenvalue)  # decrease
eigenvector_sort = eigenvector[:, eigenvalue_index_sort]

# reduce to 1D pc 1st
reduce_1D_pc1 = np.dot(original_data_mean, eigenvector_sort[:, 0])
# reduce to 1D pc 2nd
reduce_1D_pc2 = np.dot(original_data_mean, eigenvector_sort[:, 1])

# PC1 direction and component
l1, = plt.plot([0, eigenvector_sort[0][0]], [0, eigenvector_sort[1][0]], c='r')
x1_coordinates = eigenvector_sort[0][0] * reduce_1D_pc1
y1_coordinates = eigenvector_sort[1][0] * reduce_1D_pc1
p1 = plt.scatter(x1_coordinates, y1_coordinates, s=20, c='r', marker='*')

# PC2 direction and component
l2, = plt.plot([0, eigenvector_sort[0][1]], [0, eigenvector_sort[1][1]], c='g')
x2_coordinates = eigenvector_sort[0][1] * reduce_1D_pc2
y2_coordinates = eigenvector_sort[1][1] * reduce_1D_pc2
p2 = plt.scatter(x2_coordinates, y2_coordinates, s=20, c='g', marker='^')

# plot the original data
p0 = plt.scatter(original_data[:, 0], original_data[:, 1], s=10)
plt.legend(handles=[l1, l2, p0, p1, p2], labels=['first principal direction', 'second principal direction', 'data',
                                                 'first principal components', 'second principal components'])
plt.title('Linear PCA demo')
plt.show()

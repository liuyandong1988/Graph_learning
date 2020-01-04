#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Yandong
# Time  :2019/11/20 6:27
# Compress the image by PCA

import numpy as np
import pylab as pl


def princomp(A, numpc=0):
    """
    :param A:  the original data
    :param numpc: the principal component number
    :return: coeff: the main eigenvector
             score: switch the low dimension space
             latent: eigenvalue
    """
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A-np.mean(A.T, axis=1)).T  # subtract the mean (along columns)
    [latent, coeff] = np.linalg.eig(np.cov(M))
    p = np.size(coeff, axis=1)
    idx = np.argsort(latent) # sorting the eigenvalues
    idx = idx[::-1]       # in ascending order
    # sorting eigenvectors according to the sorted eigenvalues
    coeff = coeff[:, idx]
    latent = latent[idx] # sorting eigenvalues
    if 0 < numpc < p:
        coeff = coeff[:, range(numpc)] # cutting some PCs if needed
        score = np.dot(coeff.T, M) # projection of the data in the new space
    return coeff, score, latent


A = pl.imread('original.jpg')  # load an image
A = np.mean(A, 2)  # to get a 2-D array
full_pc = np.size(A, axis=1)  # numbers of all the principal components
i = 1
dist = []
for numpc in range(10, 60, 10): # 0 10 20 ... full_pc
    coeff, score, latent = princomp(A, numpc)
    Ar = np.dot(coeff, score).T + np.mean(A, axis=0)  # image reconstruction
    # difference in Frobenius norm
    dist.append(np.linalg.norm(A-Ar, 'fro'))
    # showing the pics reconstructed with less than 50 PCs
    if numpc <= 50:
        ax = pl.subplot(2, 3, i, frame_on=False)
        ax.xaxis.set_major_locator(pl.NullLocator())  # remove ticks
        ax.yaxis.set_major_locator(pl.NullLocator())
        i += 1
        pl.imshow(pl.flipud(Ar))
        pl.title('PCs # '+str(numpc))
        pl.gray()

ax = pl.subplot(2, 3, 6, frame_on=False)
ax.xaxis.set_major_locator(pl.NullLocator()) # remove ticks
ax.yaxis.set_major_locator(pl.NullLocator())
pl.imshow(pl.flipud(A))
pl.title('numpc FULL')
pl.gray()
pl.show()
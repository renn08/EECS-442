#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb


def colormapArray(X, colors):
    """
    Basically plt.imsave but return a matrix instead

    Given:
        a HxW matrix X
        a Nx3 color map of colors in [0,1] [R,G,B]
    Outputs:
        a HxW uint8 image using the given colormap
    """
    N, M = np.shape(colors)
    # print(N)

    return None


if __name__ == "__main__":
    colors = np.load("mysterydata/colors.npy")
    data = np.load("mysterydata/mysterydata3.npy")
    row, col, hei = np.shape(data)
    # for data2
    # im1 = np.sum((((data / 1.3 - data.min()) / (data.max() - data.min())) * 1023).astype(np.uint8) , axis = 2)

    # for data4
    # im1 = np.sum((((data - data.min()) / (data.max() - data.min())) * 1023).astype(np.uint8), axis=2)

    # for data3
    im1 = np.sum((((data / 1.05 - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))) * 1023).astype(np.uint8), axis=2)

    red = np.zeros((row, col))
    green = np.zeros((row, col))
    blue = np.zeros((row, col))

    for i in range(row):
        for j in range(col):
            for k in range(3):
                index = im1[i][j]
                red[i][j] = colors[index][0]
                green[i][j] = colors[index][1]
                blue[i][j] = colors[index][2]
    im = np.dstack((red, green, blue))
    imgplot = plt.imshow(im)
    plt.imsave("mysterydata3.png", im)
    plt.show()


    # pdb.set_trace()

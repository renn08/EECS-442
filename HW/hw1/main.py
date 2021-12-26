"""
Starter code for EECS 442 W21 HW1
"""
import os
import cv2
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from util import generate_gif, renderCube


def rotX(theta):
    """
    Generate 3D rotation matrix about X-axis
    Input:  theta: rotation angle about X-axis
    Output: Rotation matrix (3 x 3 array)
    """
    x1 = [1, 0, 0]
    x2 = [0, np.cos(theta), -np.sin(theta)]
    x3 = [0, np.sin(theta), np.cos(theta)]
    mat = np.vstack((x1, x2, x3))
    return mat


def rotY(theta):
    """
    Generate 3D rotation matrix about Y-axis
    Input:  theta: rotation angle along y-axis
    Output: Rotation matrix (3 x 3 array)
    """
    x1 = [np.cos(theta), 0, np.sin(theta)]
    x2 = [0, 1, 0]
    x3 = [-np.sin(theta), 0, np.cos(theta)]
    mat = np.vstack((x1, x2, x3))
    return mat


def part1():
    # TODO: Solution for Q1
    # Task 1: Use rotY() to generate cube.gif
    rotList = [rotY(0), rotY(np.pi / 3), rotY(np.pi / 3 * 2), rotY(np.pi), rotY(np.pi / 3 * 4), rotY(np.pi / 3 * 5)]
    generate_gif(rotList)

    # Task 2:  Use rotX() and rotY() sequentially to check
    # the commutative property of Rotation Matrices
    rotList2 = [rotY(np.pi/4), np.dot(rotX(np.pi/4), rotY(np.pi/4))]
    generate_gif(rotList2, "YX.gif")
    rotList3 = [rotX(np.pi/4), np.dot(rotY(np.pi/4), rotX(np.pi/4))]
    generate_gif(rotList3, "XY.gif")
    
    # Task 3: Combine rotX() and rotY() to render a cube 
    # projection such that end points of diagonal overlap
    # Hint: Try rendering the cube with multiple configrations
    # to narrow down the search region
    rotList4 = [np.dot(rotY(np.arcsin(np.sqrt(3) / 3)), rotX(np.pi/4))]
    generate_gif(rotList4, "onepoint.gif")
    pass


def split_triptych(trip):
    """
    Split a triptych into thirds
    Input:  trip: a triptych (H x W matrix)
    Output: R, G, B martices
    """
    row, col = trip.shape
    B, G, R = trip[0 : row // 3, :], trip[row // 3 : row // 3 * 2, :], trip[row // 3 * 2 : , :]
    # TODO: Split a triptych into thirds and 
    # return three channels as numpy arrays
    return R, G, B


def normalized_cross_correlation(ch1, ch2):
    """
    Calculates similarity between 2 color channels
    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
    Output: normalized cross correlation (scalar)
    """
    ch1_normed = ch1 / np.linalg.norm(ch1)
    ch2_normed = ch2 / np.linalg.norm(ch2)
    return np.sum(ch1_normed * ch2_normed)

# def metric_1(ch1, ch2, Xoffset, Yoffset):
#     row1, col1 = ch1.shape
#     ch1_normed = ch1 / np.linalg.norm(ch1)
#     ch2_normed = ch2 / np.linalg.norm(ch2)
#     # norm1 = np.linalg.norm(ch1)
#     # norm2 = np.linalg.norm(ch2)
#     # ch1_normed = ch1 / norm1
#     # ch2_normed = ch2 / norm2
#     one = np.ones((row1, col1))
#     one[0:10,:] = 0
#     one[-10:,:] = 0
#     one[:,0:10] = 0
#     one[:,-10:] = 0
    # zero_1 = np.zeros((row1, col1))
    # if Xoffset >= 0:
    #     zero[0:Xoffset,:] = 1
    #     zero_1[row1-Xoffset:,:] = 1
    # else:
    #     zero[row1+Xoffset:,:] = 1
    #     zero_1[0:-Xoffset, :] = 1
    # if Yoffset >= 0:
    #     zero[:,0:Yoffset] = 1
    # #     zero_1[:, col1 - Yoffset:] = 1
    # # else:
    # #     zero[:,col1+Yoffset:] = 1
    # #     zero_1[:, 0:-Yoffset] = 1
    # # ch2_moved_normed = ch2_normed.copy()
    # ch2_normed = np.roll(ch2_normed, Xoffset, axis=0)
    # ch2_normed = np.roll(ch2_normed, Yoffset, axis=1)
    # # return np.sum(ch1_normed * ch2_normed * zero) + np.sum(ch1_normed * (1 - zero) * ch2_moved_normed)
    # return np.sum(ch1_normed[15:-15,15:-15] * ch2_normed[15:-15,15:-15])


def best_offset(ch1, ch2, metric, Xrange=np.arange(-10, 10), 
                Yrange=np.arange(-10, 10)):
    """
    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
            metric: similarity measure between two channels
            Xrange: range to search for optimal offset in vertical direction
            Yrange: range to search for optimal offset in horizontal direction
    Output: optimal offset for X axis and optimal offset for Y axis

    Note: Searching in Xrange would mean moving in the vertical 
    axis of the image/matrix, Yrange is the horizontal axis 
    """
    # TODO: Use metric to align ch2 to ch1 and return optimal offsets
    product = metric(ch1[15:-15,15:-15], ch2[15:-15,15:-15])
    opt_vmov = 0
    opt_hmov = 0
    for vmov in Xrange:
        for hmov in Yrange:
            ch2_roll = np.roll(ch2, vmov, axis=0)
            ch2_roll = np.roll(ch2_roll, hmov, axis=1)
            product_temp = metric(ch1[15:-15, 15:-15], ch2_roll[15:-15, 15:-15])
            # ch2_normed = np.roll(ch2, vmov, axis=1)
            # ch2_normed = np.roll(ch2_normed, hmov, axis=0)
            # product_temp = metric(ch1, ch2_normed, 0, 0)
            if product_temp > product:
                product = product_temp
                opt_hmov = hmov
                opt_vmov = vmov
    return opt_vmov, opt_hmov


def align_and_combine(R, G, B, metric):
    """
    Input:  R: red channel
            G: green channel
            B: blue channel
            metric: similarity measure between two channels
    Output: aligned RGB image 
    """
    # TODO: Use metric to align the three channels 
    # Hint: Use one channel as the anchor to align other two
    opt_G_v, opt_G_h = best_offset(R, G, metric)
    opt_B_v, opt_B_h = best_offset(R, B, metric)
    G = np.roll(G, opt_G_v, axis=0)
    G = np.roll(G, opt_G_h, axis=1)
    B = np.roll(B, opt_B_v, axis=0)
    B = np.roll(B, opt_B_h, axis=1)
    img = np.dstack((R, G, B))
    return img


def pyramid_align(filename):
    # TODO: Reuse the functions from task 2 to perform the 
    # image pyramid alignment iteratively or recursively
    img = plt.imread("tableau/" + filename + ".jpg")
    R, G, B = split_triptych(img)
    row, col = R.shape
    # level-0
    dim_0 = (col // 16, row // 16)
    R_0 = cv2.resize(R, dim_0, interpolation=cv2.INTER_AREA)
    G_0 = cv2.resize(G, dim_0, interpolation=cv2.INTER_AREA)
    B_0 = cv2.resize(B, dim_0, interpolation=cv2.INTER_AREA)
    plt.imsave(filename + "_0_origin.jpg", np.dstack((R_0, G_0, B_0)))
    img = align_and_combine(R_0, G_0, B_0, normalized_cross_correlation)
    plt.imsave(filename + "_0.jpg", img)
    best_G_0 = best_offset(R_0, G_0, normalized_cross_correlation)
    best_B_0 = best_offset(R_0, B_0, normalized_cross_correlation)
    # level-1
    dim_1 = (col // 4, row // 4)
    R_1 = cv2.resize(R, dim_1, interpolation=cv2.INTER_AREA)
    G_1 = cv2.resize(G, dim_1, interpolation=cv2.INTER_AREA)
    B_1 = cv2.resize(B, dim_1, interpolation=cv2.INTER_AREA)
    plt.imsave(filename + "_1_origin.jpg", np.dstack((R_1, G_1, B_1)))
    G_1 = np.roll(G_1, best_G_0[0] * 4, axis=0)
    G_1 = np.roll(G_1, best_G_0[1] * 4, axis=1)
    B_1 = np.roll(B_1, best_B_0[0] * 4, axis=0)
    B_1 = np.roll(B_1, best_B_0[1] * 4, axis=1)
    img = align_and_combine(R_1, G_1, B_1, normalized_cross_correlation)
    plt.imsave(filename + "_1.jpg", img)
    best_G_1 = best_offset(R_1, G_1, normalized_cross_correlation)
    best_B_1 = best_offset(R_1, B_1, normalized_cross_correlation)
    # level-2
    dim_2 = (col, row)
    R_2 = cv2.resize(R, dim_2, interpolation=cv2.INTER_AREA)
    G_2 = cv2.resize(G, dim_2, interpolation=cv2.INTER_AREA)
    B_2 = cv2.resize(B, dim_2, interpolation=cv2.INTER_AREA)
    plt.imsave(filename + "_2_origin.jpg", np.dstack((R_2, G_2, B_2)))
    G_2 = np.roll(G_2, best_G_0[0] * 16 + best_G_1[0] * 4, axis=0)
    G_2 = np.roll(G_2, best_G_0[1] * 16 + best_G_1[1] * 4, axis=1)
    B_2 = np.roll(B_2, best_B_0[0] * 16 + best_B_1[0] * 4, axis=0)
    B_2 = np.roll(B_2, best_B_0[1] * 16 + best_B_1[1] * 4, axis=1)
    img = align_and_combine(R_2, G_2, B_2, normalized_cross_correlation)
    plt.imsave(filename + "_2.jpg", img)
    pass


def part2():
    # TODO: Solution for Q2
    # Task 1: Generate a colour image by splitting 
    # the triptych image and save it
    img = plt.imread("prokudin-gorskii/00153v.jpg")
    R, G, B = split_triptych(img)
    R = R[0:-1,:]
    img_new = np.dstack((R, G, B))
    plt.imsave("00153v_color.jpg", img_new)

    # Task 2: Remove misalignment in the colour channels 
    # by calculating best offset
    # img = plt.imread("tableau/efros_tableau.jpg")
    # R, G, B = split_triptych(img)
    img_aligned = align_and_combine(R, G, B, normalized_cross_correlation)
    plt.imsave("00153v_color_aligned.jpg", img_aligned)
    # Task 3: Pyramid alignment
    pass


def part3():
    # TODO: Solution for Q3
    pyramid_align("seoul_tableau")
    pyramid_align("vancouver_tableau")
    pass

def part4():
    # img0 = plt.imread("Mhat0.jpeg")
    # img1 = plt.imread("Mhat1.jpeg")
    # R0, G0, B0 = img0[:,:,0], img0[:,:,1], img0[:,:,2]
    # plt.imsave("R0.jpg",R0)
    # plt.imsave("G0.jpg",G0)
    # plt.imsave("B0.jpg",B0)
    # R1, G1, B1 = img1[:, :, 0], img1[:, :, 1], img1[:, :, 2]
    # plt.imsave("R1.jpg", R1)
    # plt.imsave("G1.jpg", G1)
    # plt.imsave("B1.jpg", B1)
    # pass

def main():
    part1()
    part2()
    part3()
    # part4()


if __name__ == "__main__":
    main()

from scipy import ndimage, misc
import math
import numpy as np
import utils
import cv2

import matplotlib.pyplot as mp

def rotate(features, alpha):
    Y = misc.imrotate(X, 15)
    lenx1, lenx2 = X.shape;
    leny1, leny2 = Y.shape;

    #  Trim the result back to the original size, around its center point.
    fromx = math.floor((leny1 + 1 - lenx1)/2);
    fromy = math.floor((leny2 + 1 - lenx2)/2);

    destination = Y[fromx:fromx+lenx1, fromy:fromy+lenx2]

    img_binary = cv2.threshold(destination, 128,255, cv2.THRESH_BINARY)[1]

    rows,col=np.where(img_binary == 255)
    img_binary[rows, col] = 1

    return img_binary


if __name__ == '__main__':
    train_data = utils.read_data_seq('../data/train_mini.txt')

    ex = train_data[0]


    # print(ex)
    ex_pic, ex_letter = ex[1][1].reshape(16,8), ex[0][1]
    X = ex_pic

    print(ex_pic, ex_letter)
    mp.figure(1)
    mp.imshow(ex_pic, cmap='gray')


    img_binary = rotate(X, 15)


    mp.figure(2)
    mp.imshow(img_binary, cmap='gray')
    mp.show()




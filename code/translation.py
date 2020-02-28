import cv2
import numpy as np
import utils,string
import matplotlib.pyplot as mp
#, compute, check_grad

def read_data():
    train_data = utils.read_data_seq('../data/train_mini.txt')
    trainX, trainY = [], []
    for d in train_data:
        trainX.append(d[1])
        trainY.append(d[0])

    return trainX, trainY

#offset (tx, ty)
def translate_data(features, tx, ty):
    
    lenX, lenY = features.shape
    translated = np.zeros((lenX, lenY), features.dtype)

    M = np.float32([[1,0,tx],[0,1,ty]])
    dst = cv2.warpAffine(features,M,(lenY,lenX))

    mp.imshow(dst, cmap='gray')
    mp.show()

	
if __name__ == "__main__":
    X, Y = read_data()	
    #train_data = utils.read_data_seq('../data/train_mini.txt') 

    #p, letter = train_data[1], train_data[0]
    pic = X[0][1]

    print(np.array(pic).shape)
    
    pic_d = np.array(pic).reshape(16,8)
    print(pic_d.shape)

    #original figure
    #mp.figure(3)
    mp.imshow(pic_d, cmap = 'gray')
    mp.show()
    #translation
    translate_data(pic_d,3,3)

    #lists of transformations from translation.txt (haven't done yet)
    f = open('../data/translation_mini.txt', 'r')

    for line in f:
        offset = []
        content = line.rstrip().split(' ')
        flag, nth, offx, offy = content[0], int(content[1]), int(content[2]), int(content[3])
        #if(flag=='t'):
            #print('word count: ', nth, '(', offx, offy, ')')
            #print('origin data:\n', X[nth])
            #print(np.array(X[nth]).shape)
            #translate_data(pic[nth], offx, offy)



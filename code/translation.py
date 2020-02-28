import cv2
import numpy as np
import utils,string
import matplotlib
#, compute, check_grad

def read_data():
    train_data = utils.read_data_seq('../data/train_mini.txt')
    trainX, trainY = [], []
    for d in train_data:
        trainX.append(d[1])
        trainY.append(d[0])
    print(np.array(trainX).shape)
    print(np.array(trainY).shape)

    return trainX, trainY

#offset (tx, ty)
def translate_data(data, tx, ty):
    
    lenX, lenY = data.shape
    translated = np.zeros((lenX, lenY), data.dtype)
    

    #print('translated:\n', translated)

    #img = cv2.imread('messi5.jpg',0)
    #rows,cols = data.shape

    M = np.float32([[1,0,tx],[0,1,ty]])
    dst = cv2.warpAffine(img,M,(lenY,lenX))

    mp.figure(2)
    mp.imshow(dst, cmap='gray')
    mp.show()

    cv2.imshow('img',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pic(data):
    img = numpy.zeros([16,8,3])

    img[:,:,0] = numpy.ones([16,8])*64/255.0
    img[:,:,1] = numpy.ones([16,8])*128/255.0
    img[:,:,2] = numpy.ones([16,8])*192/255.0

    cv2.imwrite('../color_img.jpg', img)
    cv2.imshow("image", img);
    cv2.waitKey();
	
if __name__ == "__main__":
    X, Y = read_data()	
    #train_data = utils.read_data_seq('../data/train_mini.txt') 
    #pic, letter = train_data[1], train_data[0]
   
    #print(pic)
    f = open('../data/translation_mini.txt', 'r')


    #data = np.array(X).reshape(16,8)
    for line in f:
        offset = []
        content = line.rstrip().split(' ')
        flag, nth, offx, offy = content[0], int(content[1]), int(content[2]), int(content[3])
        if(flag=='t'):
            print('word count: ', nth, '(', offx, offy, ')')
            print('origin data:\n', X[nth])
            print(np.array(X[nth]).shape)
            #translate_data(pic[nth], offx, offy)
        #for x, y in zip(X,Y):


import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# #PRACTICING HERE    #PRACTICING HERE    #PRACTICING HERE    #PRACTICING HERE    #PRACTICING HERE    #PRACTICING HERE    #PRACTICING HERE    
# training_data = open("train.txt","r").readlines() #grab training data

# # gets a specific line to test on, with actual letter "k"
# line_number = 4
# specific_line = training_data[line_number].strip("\n").split(" ")
# features = specific_line[5:]

# # gets an image of the specific line we are interested in 
# total_features = []
# for i in range(0,16):
#     feature_line = []
#     for j in range(0,8):
#         feature_line.append(int(features[8*i+j]))
#     total_features.append(feature_line)

# #turn total features to an np.array
# total_features = np.array(total_features)

# file_name = "image.jpg" 
# plt.imsave(file_name,total_features, cmap="Greys") #save the features as an image
# img = cv2.imread(file_name, 0) #read the image back in grey scale
# img_binary = cv2.threshold(img, 128,255, cv2.THRESH_BINARY)[1]
# cols,rows = img.shape

# #this shows image "img"
# plt.imshow(img_binary)
# plt.show()

# degree = 15 #degree we want to rotate and test by
# matrix = cv2.getRotationMatrix2D((rows/2, cols/2),degree,1) #get the rotational matrix
# destination = cv2.warpAffine(img_binary, matrix, (rows,cols)) #apply the rotation matrix to the image

# # plt.imshow(destination) #sets up the picture to image
# # plt.show() #shows the new image

# # print(img)
# # print(total_features)
# # print(img_binary)
# # print(destination)


# #turns destination into a 0/1 2d array but captures the rotation
# new_destination = [] #our final 2d array 
# for row in destination:
#     new_row = []
#     for val in row:
#         if val == 255:
#             new_row.append(1)
#         else:
#             new_row.append(0)
#     new_destination.append(new_row)

# new_destination = np.array(new_destination) #turn it into an np.array

# # plt.imshow(new_destination) #sets up the picture to image
# # plt.show() #shows the new image

# #PRACTICING HERE    #PRACTICING HERE    #PRACTICING HERE    #PRACTICING HERE    #PRACTICING HERE    #PRACTICING HERE    

# ________________________________________________________________________________________________________________________
#takes in the transform file, and a list
#returns the transform text in the form of a list that is ordered by the word number
def sort_transform_txt(transform_text, to_return_ordered):
    transform_text = open(transform_text,"r").readlines()  

    for curr_line in transform_text:
        line = curr_line.strip("\n").split(" ")
        transformation = line[0]
        word_no = int(line[1])

        #if rotate, get angle, if translate, get both values
        if transformation == 'r':
            displacement = line[2]
        elif transformation == 't':
            displacement = [line[2],line[3]]


        to_return_ordered.append([word_no, transformation, displacement])
        to_return_ordered.sort()

#this is the actual sorting of the transform
sorted_transform_text = []
sort_transform_txt("transform.txt", sorted_transform_text)

#proof this is working
# print(sorted_transform_text[:3])
# ________________________________________________________________________________________________________________________

if os.path.exists('new_train.txt'):
    os.remove('new_train.txt')
#this tranforms the training data

#for the rotate...
#goes through the transform.txt one line at a time, because each line is a word number 
#then goes through train.txt and does a word number match, so it rotates every single letter in that word
with open("new_train.txt","w") as new_training_data:


    training_data = open("train.txt","r").readlines() #read the training data in
    training_line = 0 #we start at line 0 of the data

    for curr_line in sorted_transform_text: #iterate through all the necessary transforms
    # for i in range(0,2):
        # curr_line = sorted_transform_text[i] #current word we are interested in transforming
        word_no = curr_line[0] #get the word number from transform text
        how_to_transform = curr_line[1]

        #check the current line we are at in train.txt
        training_curr_line = training_data[training_line].strip("\n").split(" ")
        training_retain_this = training_curr_line[0:4]
        training_word_no = training_curr_line[3]

        #if the word number matches our number, we want to transform or rotate that letter
        #then we + 1 on the training_line and repeat until it is not true
        while int(training_word_no) == int(word_no):
            features = training_curr_line[5:] #grab our features from the current line in train
                 
          #if we are doing a rotation...
            if how_to_transform == 'r':
                rotation_degrees = int(curr_line[2]) #this is how much we have to rotate by
                total_features = [] #capture the features as a 2d array

                for i in range(0,16):
                    feature_line = []
                    for j in range(0,8):
                        feature_line.append(int(features[8*i+j]))
                    total_features.append(feature_line)

                file_name = "image.jpg"  #file name of image to be saved
                plt.imsave(file_name,total_features, cmap="Greys") #save the features as a grey image
                img = cv2.imread(file_name, 0) #read the image back in grey scale
                img_binary = cv2.threshold(img, 128,255, cv2.THRESH_BINARY)[1] #new image that is only 0/255
                cols,rows = img_binary.shape

                matrix = cv2.getRotationMatrix2D((rows/2, cols/2),rotation_degrees,1) #get the rotational matrix
                destination = cv2.warpAffine(img_binary, matrix, (rows,cols)) #apply the rotation matrix to the image

                #turns destination into a 0/1 2d array but captures the rotation
                new_destination = [] #our final 2d array 
                for row in destination:
                    new_row = []
                    for val in row:
                        if val == 255:
                            new_row.append(1)
                        else:
                            new_row.append(0)
                    new_destination.append(new_row)

                new_destination = np.array(new_destination) #turn it into an np.array

                new_whole_row = training_retain_this
                for row in new_destination: #we add onto the retain_this part so we get a whole list of str to write in
                    for val in row:
                        new_whole_row.append(str(val))

                #add it to the new training file
                new_whole_row = " ".join(new_whole_row) + '\n'
                new_training_data.write(new_whole_row)

            training_line +=1    #go to the next line in the train.txt file
            if training_line >= len(training_data): #if we reached the end of the file, don't do the next stuff
                break
            training_curr_line = training_data[training_line].strip("\n").split(" ")
            training_retain_this = training_curr_line[0:4]
            training_word_no = training_curr_line[3]
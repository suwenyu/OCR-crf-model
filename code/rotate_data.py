import cv2
import matplotlib.pyplot as plt

#this rotates the TRAINING DATA
training_data = open("train.txt","r").readlines()

# gets a specific line
line_number = 1
specific_line = training_data[line_number].strip("\n").split(" ")
features = specific_line[5:]

# gets an image of the specific line we are interested in 
total_features = []
for i in range(0,16):
    feature_line = []
    for j in range(0,8):
        feature_line.append(int(features[8*i+j]))
    total_features.append(feature_line)

#show the original image
# plt.matshow(total_features)
# plt.show()

#APPLY ROTATION
file_name = "image.jpg"
plt.imsave(file_name,total_features, cmap="Greys")
img = cv2.imread(file_name)
cols,rows,ch = img.shape

degree = 180
matrix = cv2.getRotationMatrix2D((rows/2, cols/2),degree,1)
destination = cv2.warpAffine(img, matrix, (rows,cols))

# plt.imsave(file_name, destination)
plt.imshow(destination)
# plt.show()

print(img.shape)

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

sorted_transform_text = []
sort_transform_txt("transform.txt", sorted_transform_text)

#proof this is working
# print(sorted_transform_text[:3])

# ________________________________________________________________________________________________________________________
#this tranforms the training data
training_data = open("train.txt","r").readlines()
training_line = 0

# for curr_line in sorted_transform_text:
#     word_no = curr_line[0]

#     #check the current line we are at in train.txt
#     training_curr_line = training_data[training_line].strip("\n").split(" ")
#     training_word_no = training_curr_line[3]

#     #if the word number matches our number, we want to transform or rotate that letter
#     #then we + 1 on the training_line and repeat until it is not true
#     while training_word_no == word_no:
#         features = training_curr_line[5:]
#         training_line +=1

# #       #if we are doing a rotation...
#         if curr_line[1] == 'r':

#             rotation_degrees = int(curr_line[2])
#             total_features = []

#             for i in range(0,16):
#                 feature_line = []
#                 for j in range(0,8):
#                     feature_line.append(int(features[8*i+j]))
#                 total_features.append(feature_line)

#             file_name = "image.jpg"
#             plt.imsave(file_name,total_features)
#             img = cv2.imread(file_name)
#             cols,rows,ch = img.shape

#             matrix = cv2.getRotationMatrix2D((rows/2, cols/2),rotation_degrees,1)
#             destination = cv2.warpAffine(img, matrix, (rows,cols))
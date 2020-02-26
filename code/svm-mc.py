import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
import lib_linear.python.liblinearutil as liblinear

training_data = open("train.txt","r").readlines()
testing_data = open("test.txt","r").readlines()

train_y = []
train_x = []
test_y = []
test_x = []
test_next_id_list = []

#TRAINING
for i in range(0, len(training_data)):
    input_line = training_data[i].strip("\n").split(" ")

    #get id, y, and features/x
    # input_letter_id = input_line[0]
    input_alphabet = input_line[1]
    # input_next_id = input_line[2]
    # input_word_id = input_line[3]
    input_features = input_line[4:]
    
    temp = [int(i) for i in input_features]
    input_features = temp

    train_y.append(input_alphabet)
    train_x.append(input_features)    


#TESTING
for i in range(0, len(testing_data)):
    testing_line = testing_data[i].strip("\n").split(" ")    
    
    #get id, y, and features/x
    # testing_letter_id = testing_line[0]
    testing_alphabet = testing_line[1]
    testing_next_id = testing_line[2]
    testing_word_id = testing_line[3]
    testing_features = testing_line[4:]
    
    temp = [int(i) for i in testing_features]
    testing_features = temp
    
    test_y.append(testing_alphabet)
    test_x.append(testing_features)    
    test_next_id_list.append(testing_next_id)

c_val = 1

# while (c_val < 20000):
#this will need to be done after updating values of C, may need to update max_iter
model = LinearSVC(max_iter = 1000, C = c_val).fit(train_x,train_y)

letter_wise_accuracy = 0
letter_wise_total = 0

word_wise_accuracy = 0
word_wise_total = 0

word_accurate_bool = 1

for i in range(0, len(testing_data)):
    prediction = model.predict([test_x[i]])

    if str(test_y[i]) == prediction:
        letter_wise_accuracy +=1
    else:
        word_accurate_bool = 0
    letter_wise_total +=1

    if int(test_next_id_list[i]) == -1:
        word_wise_accuracy += word_accurate_bool
        word_wise_total +=1
        word_accurate_bool = 1

    # print(test_y[i], ":", prediction)    

letter_accuracy = float(letter_wise_accuracy/letter_wise_total)
word_accuracy = float(word_wise_accuracy/word_wise_total)
print(letter_accuracy)
print(word_accuracy)
    

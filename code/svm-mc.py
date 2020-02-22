import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC

training_data = open("train.txt","r").readlines()
testing_data = open("test.txt","r").readlines()

train_y = []
train_x = []
test_y = []
test_x = []

for i in range(0, len(testing_data)):
    #TESTING
    testing_line = testing_data[i].strip("\n").split(" ")    
    
    #get id, y, and features/x
    testing_letter_id = testing_line[0]
    testing_alphabet = testing_line[1]
    testing_next_id = testing_line[2]
    testing_word_id = testing_line[3]
    testing_features = testing_line[4:]  
    
    temp = [int(i) for i in testing_features]
    testing_features = temp
    
    test_y.append(testing_alphabet)
    test_x.append(testing_features)    

for i in range(0, len(training_data)):
    #TRAINING 
    input_line = training_data[i].strip("\n").split(" ")

    #get id, y, and features/x
    input_letter_id = input_line[0]
    input_alphabet = input_line[1]
    input_next_id = input_line[2]
    input_word_id = input_line[3]
    input_features = input_line[4:]
    
    temp = [int(i) for i in input_features]
    input_features = temp

    train_y.append(input_alphabet)
    train_x.append(input_features)    
    
c_val = 1
c_val_list = []
score_list = []

while (c_val < 200000):
    model = LinearSVC(max_iter = 10000, c = c_val).fit(train_x,train_y)
    score = model.score(test_x,test_y)

    c_val_list.append(c_val)
    score_list.append(score)

    c_val = c_val * 2
    print("score with c_val {}:".format(c_val), score)

    
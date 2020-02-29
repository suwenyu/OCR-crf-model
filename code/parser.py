import subprocess
import os
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
#subprocess() to execute the training and validation commands

#svm_hmm

letter_wise_accuracy_hmm = []
word_wise_accuracy_hmm = []
c_hmm = []
c = 1
while(c < 200000):
  subprocess.call([r"./../data/svm_hmm/svm_hmm_learn", "-c", str(c), "./../data/train_struct.txt"])
  subprocess.call([r"./../data/svm_hmm/svm_hmm_classify", "./../data/test_struct.txt", "./svm_struct_model", "test.outtags"])
  myDoc = open("test.outtags", "r").read()
  #compare test.outtags with the first coloumn of test_stuct.txt for accuracy

  test_struct_txt = open("../data/test_struct.txt", 'r').readlines() 
  test_outtags = open("test.outtags","r").readlines()

  letter_wise_correct = 0
  letter_wise_total = 0

  word_wise_accuracy = 0
  word_wise_total = 0
  qid_counter = 1
  word_accurate_bool = 1

  for i in range(0,len(test_struct_txt)):
    specific_line = test_struct_txt[i]

    splitter = specific_line.split("qid:") #splits the line by qid
    alphabet = splitter[0].strip(" ") #get the number in alphabet which corresponds to character
    qid = splitter[1].split(" ")[0] #get the qid to know if it is the same word

    #wordwise accuracy
    if (int(qid) == qid_counter):
      pass    #we reset the boolean and assume true 
    else:
      #if we changed to a new word, we check if we had the word right
      qid_counter +=1
      word_wise_accuracy += word_accurate_bool
      word_wise_total += 1
      word_accurate_bool = 1

    #this is for each line in test_outtags
    outtag_alphabet = test_outtags[i].strip(" ").strip("\n")

    #get letterwise accuracy
    if (outtag_alphabet == alphabet):
      letter_wise_correct +=1
    else:
      word_accurate_bool = 0
    #increase letter count by 1
    letter_wise_total +=1

  letterAccuracy = float(letter_wise_correct)/float(letter_wise_total)
  wordAccuracy = float(word_wise_accuracy)/float(word_wise_total)

  letter_wise_accuracy_hmm.append(letterAccuracy)
  word_wise_accuracy_hmm.append(wordAccuracy)
  c_hmm.append(c)
  print('Iteration with c={} complete'.format(c))
  c = c*2
#plot(x=c_hmm, y=y_hmm)

for i in range(0, len(c_hmm)):
  print('C = {}: letter wise accuracy = {}, word wise accuracy = {}'.format(c_hmm[i], letter_wise_accuracy_hmm[i], word_wise_accuracy_hmm[i]))


#######LIBLINEAR SVC NEXT#######

training_data = open("../data/train.txt","r").readlines()
testing_data = open("../data/test.txt","r").readlines()

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

c_val = .125
c_val_liblinear = []
letter_wise_liblinear = []
word_wise_liblinear = []

while (c_val <= 512):
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

  letter_accuracy = float(letter_wise_accuracy)/float(letter_wise_total)
  word_accuracy = float(word_wise_accuracy)/float(word_wise_total)
  letter_wise_liblinear.append(letter_accuracy)
  word_wise_liblinear.append(word_accuracy)
  c_val_liblinear.append(c_val)
  c_val *= 2


#######LIBLINEAR SVC ABOVE#######

#SVM_HMM letter-wise
plt.plot(c_hmm, letter_wise_accuracy_hmm)
plt.ylabel('letter-wise accuracy (SVM_hmm)')
plt.xlabel('C-Value (SVM_hmm)')
plt.xscale('log')
plt.show()


#SVM_HMM word-wise
plt.plot(c_hmm, word_wise_accuracy_hmm)
plt.ylabel('word-wise accuracy (SVM_hmm)')
plt.xlabel('C-Value (SVM_hmm)')
plt.xscale('log')
plt.show()


#LibLinear letter-wise
plt.plot(c_val_liblinear, letter_wise_liblinear)
plt.ylabel('letter-wise accuracy (LibLinear)')
plt.xlabel('C-Value (LibLinear)')
plt.xscale('log')
plt.show()


#LibLinear word-wise
plt.plot(c_val_liblinear, word_wise_liblinear)
plt.ylabel('word-wise accuracy (LibLinear)')
plt.xlabel('C-Value (LibLinear)')
plt.xscale('log')
plt.show()


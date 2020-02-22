import subprocess
import os
from sklearn.svm import LinearSVC
#subprocess() to execute the training and validation commands

#svm_hmm
letter_wise_accuracy_hmm = []
word_wise_accuracy_hmm = []
c_hmm = []
c = 1
while(c < 200000):
  subprocess.call([r"./svm_hmm/svm_hmm_learn", "-c", str(c), "train_struct.txt"])
  subprocess.call([r"./svm_hmm/svm_hmm_classify", "test_struct.txt", "svm_struct_model", "test.outtags"])
  myDoc = open("test.outtags", "r").read()
  #compare test.outtags with the first coloumn of test_stuct.txt for accuracy

  test_struct_txt = open("test_struct.txt", 'r').readlines() 
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




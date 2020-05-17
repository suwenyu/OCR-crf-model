# Graphical Model for Optical Character Recognition Problem

### Description

We implemented a conditional random field for optical character recognition (OCR), with emphasis on inference and performance test. 


### Dataset
The original dataset is downloaded from http://www.seas.upenn.edu/∼taskar/ocr. It contains the image and label of 6,877 words collected from 150 human subjects, with 52,152 letters in total. To simplify feature engineering, each letter image is encoded by a 128 (=16*8) dimensional vector, whose entries are either 0 (black) or 1 (white).


### Environments and Required Packages
```bash
$ python3 --version
> version >= 3.5

$ pip3 install numpy scipy
$ pip3 install -U scikit-learn
```

### Run the program
Under the code folder
```bash
cd /path/to/assign/code/
```

### Assignment 1
##### 1(c)

1. brute force solution
```bash
$ python3 brute_force_solution.py

> [16, 5, 3]
```
2. max sum algorithm solution
```bash
$ python3 max_sum_solution.py
```
It would write "decode-output.txt" file under the result folder
```bash
$ vim ../result/decode-output.txt
```

### Assignment 2
##### 2(a)
```bash
$ python3 check_grad.py

> Gradient Value: 1.50211935303644e-05 (10 seqs)
```
It would record the w and t into "gradient.txt" file under the result folder
```bash
$ vim ../result/graduent.txt
``` 

##### 2(b)
```bash
$ python3 ref_optimize.py

> Word Accuracy = 0.46932247746437916 , Letter Accuracy = 0.8345675242384915.
```
It would store the params into the "solution.txt" file and also make the prediction of the test.txt under the "prediction.txt" file

#### 3(a) and 3(b)
```bash
$ python3 parser.py
```
Running parser.py file will run both svm-hmm and svm-mc with varying c values, with doubling powers of 2. Additionally, parser.py will print out related graphs, which can be saved. Runtime is approximately 30 minutes for each svm model.



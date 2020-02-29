# aml2020s_assign1 - Graphical Models

### Description

In this part, you will implement a conditional random field for optical character recognition (OCR), with emphasis on inference and performance test.


### Dataset
The original dataset is downloaded from http://www.seas.upenn.edu/∼taskar/ocr. 


### Environments and Required Packages
```bash
$ python3 --version
> version >= 3.5

$ pip3 install numpy scipy
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


##### Todo List
- [X] 1(a)
- [X] 1(b)
- [X] 1(c) write brute force and dp decoder.

- [X] 2(a) check w and t gradient, and some dp.
- [X] 2(b) train the crf (call 2(a) func)
- [X] 2 save params and w, t into files

- [X] 3(a) write the plot func(given X, test-acc, word-acc)
- [X] 3(a) plot svm-mc, svm-hmm, crf
- [X] 3(b) produce another three plots for word-wise prediction

- [X] 4(a) write rotate and translation (crf, and svm)
- [X] 4(b) robustness, and plot them


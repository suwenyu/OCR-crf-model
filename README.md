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

> Word Accuracy = 0.0956673451584763, Letter Accuracy = 0.5824872127643331.
```
It would store the params into the "solution.txt" file and also make the prediction of the test.txt under the "prediction.txt" file

#### 3(a) and 3(b)
```bash
$ python3 parser.py
```
Running parser.py file will run both svm-hmm and svm-mc with varying c values, with doubling powers of 2. Additionally, parser.py will print out related graphs, which can be saved. Runtime is approximately 30 minutes for each svm model.


##### Todo List
- [ ] 1(a)
- [ ] 1(b)
- [X] 1(c) write brute force and dp decoder.

- [X] 2(a) check w and t gradient, and some dp.
- [X] 2(b) train the crf (call 2(a) func)
- [X] 2 save params and w, t into files

- [x] 3(a) write the plot func(given X, test-acc, word-acc)
- [x] 3(a) plot svm-mc, svm-hmm, crf
- [x] 3(b) produce another three plots for word-wise prediction

- [x] 4(a) write rotate and translation (crf, and svm)
- [ ] 4(b) robustness, and plot them


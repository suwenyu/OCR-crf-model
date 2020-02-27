import numpy as np
import crf, utils, compute, check_grad
import time

import max_sum_solution
from scipy.optimize import fmin_bfgs
import scipy.optimize as opt

def read_data():
    train_data = utils.read_data_seq('../data/train_mini.txt')
    trainX, trainY = [], []
    for d in train_data:
        trainX.append(d[1])
        trainY.append(d[0])

    test_data = utils.read_data_seq('../data/test.txt')
    testX, testY = [], []
    for d in test_data:
        testX.append(d[1])
        testY.append(d[0])

    params = utils.load_model_params('../data/model.txt')
    # train_data = (trainX, trainY)
    # test_data = (testX, testY)

    return train_data, test_data, params


def func(params, data, c):
    n = len(data)
    l2 = 1 / 2 * np.sum(params ** 2)
    log_loss = check_grad.compute_log_p_avg(params, data, n)
    return -c * log_loss + l2

def func_prime(params, data, c):
    n = len(data)
    loss_gradient = check_grad.gradient_avg(params, data, n)
    l2_gradient = params
    return -c * loss_gradient + l2_gradient


def ref_optimize(train_data, test_data, c, params):
    print('Training CRF ... c = {} \n'.format(c))
    x0 = np.zeros((128*26+26**2,1))

    start = time.time()

    # X, y = train_data[1], train_data[0]
    result = opt.fmin_tnc(func, params, fprime=func_prime, args = [train_data, c], maxfun=100,
                          ftol=1e-3, disp=5)
    model  = result[0]
    # out = fmin_bfgs(func, params, func_prime, (train_data, c), disp=1)
    # print(model.shape)

    print("Total time: ", end='')
    print(time.time() - start)

    with open("../result/" + 'solution' + ".txt", "w") as text_file:
        for i in model:
            text_file.write(str(i) + "\n")



def decode_test_data(test_data, W, T):
    print("test the model ...")
    preds = []
    for i in test_data:
        # print(i[1])
        pred = max_sum_solution.max_sum(i[1], W, T)
        preds.append(pred)
    
    return preds


def test_model(test_data):
    params = utils.load_model_params('../result/solution.txt')
    W = utils.extract_w(params)
    T = utils.extract_t(params)

    y_preds = decode_test_data(test_data, W, T)

    f = open('../result/prediction.txt', 'w')
    for pred in y_preds:
        for word in pred:
            f.write(str(word+1) + "\n")


if __name__ == '__main__':
    c = 1000

    train_data, test_data, params = read_data()
    ref_optimize(train_data, test_data, c, params)

    test_model(test_data)




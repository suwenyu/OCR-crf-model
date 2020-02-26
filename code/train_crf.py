import numpy as np
import crf, utils, compute, check_grad
import time

from scipy.optimize import fmin_bfgs

def read_data():
    train_data = utils.read_data_seq('../data/train.txt')
    trainX, trainY = [], []
    for d in train_data:
        trainX.append(d[1])
        trainY.append(d[0])

    test_data = utils.read_data_seq('../data/test.txt')
    testX, testY = [], []
    for d in test_data:
        testX.append(d[1])
        testY.append(d[0])

    params = utils.load_model_params()
    # train_data = (trainX, trainY)
    # test_data = (testX, testY)

    return train_data, test_data, params


def func(params, data, c):
    n = len(data)
    l2 = 1 / 2 * np.sum(params ** 2)
    log_loss = check_grad.compute_log_p_avg(params, data, n)
    return -C * log_loss + l2_regularization

def func_prime(params, data, c):
    n = len(data)
    loss_gradient = check_grad.gradient_avg(params, data, n)
    l2_gradient = params
    return -C * loss_gradient + l2_gradient


def ref_optimize(train_data, test_data, c, params):
    print('Training CRF ... c = {} \n'.format(c))
    start = time.time()

    X, y = train_data[0], train_data[1]

    out = fmin_bfgs(func, params, func_prime, (train_data, c), disp=1)
    print("Total time: ", end='')
    print(time.time() - start)

    with open("result/" + 'solution' + ".txt", "w") as text_file:
        for i in out:
            text_file.write(str(i) + "\n")


if __name__ == '__main__':
    c = 1000

    train_data, test_data, params = read_data()
    ref_optimize(train_data, test_data, c, params)




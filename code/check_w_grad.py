import numpy as np
import crf, utils

from scipy.optimize import check_grad

data = utils.read_data_seq('../data/train.txt')
W, T = utils.read_model()

# print(data[0])
# print(W.shape, T.shape)
np.random.seed(0)
x0 = np.random.rand(26*128)

# print(W.shape)

def func(W, *args):
    W = W.reshape(26, 128)

    X, Y, T = args[0], args[1], args[2]
    crf_model = crf.crf(X, Y, W, T)
    return crf_model.log_prob()

def func_prime(W, *args):
    W = W.reshape(26, 128)

    X, Y, T = args[0], args[1], args[2]
    return crf.w_grad(X, Y, W, T).reshape(26*128)


print(check_grad(func, func_prime, x0, data[0][1], data[0][0], T))



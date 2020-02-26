import numpy as np
import crf, utils, compute

from scipy.optimize import check_grad

def compute_log_p_avg(params, data, iterator):
    total = 0
    W = utils.extract_w(params)
    T = utils.extract_t(params)

    for i in range(iterator):
        crf_model = crf.crf(data[i][1], data[i][0], W, T)
        total += crf_model.compute_log_prob()

    # print(total)
    return total / (iterator)


def grad_word(data, W, T, i):
    crf_model = crf.crf(data[i][1], data[i][0], W, T)
    alpha, tmp, message = crf_model.forward()
    beta, tmp, message = crf_model.backward()

    denom = crf_model.compute_z(alpha)

    w_grad = crf.w_grad(data[i][1], data[i][0], W, T, denom)
    t_grad = crf.t_grad(data[i][1], data[i][0], W, T, denom)

    # print(w_grad.shape, t_grad.shape)
    return np.concatenate((w_grad.flatten(), t_grad.flatten()))


def gradient_avg(params, data, iterator):
    total = np.zeros(128 * 26 + 26 * 26)
    W = utils.extract_w(params)
    T = utils.extract_t(params)

    for i in range(iterator):
        total += grad_word(data, W, T, i)
    
    # for i in total:
    #     print(i)
    return total / (iterator)


def check_gradient(data, params):
    # print(gradient_avg(W, T, data, 1))
    grad_value = check_grad(compute_log_p_avg, gradient_avg, params, data, 10)
    print("Gradient Value:", grad_value)


if __name__ == "__main__":

    data = utils.read_data_seq('../data/train.txt')
    params = utils.load_model_params()
    # print(params.shape)

    # crf_model = crf.crf(X, Y, W, T)
    print("check gradient ... ")
    check_gradient(data, params)
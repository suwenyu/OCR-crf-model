import numpy as np
import crf, utils, compute

from scipy.optimize import check_grad

def compute_log_p_avg(params, data, iterator):
    total = 0
    W = compute.matricize_W(params)
    T = compute.matricize_Tij(params)

    for i in range(iterator):
        crf_model = crf.crf(data[i][1], data[i][0], W, T)
        total += crf_model.log_prob()

    # print(total)
    return total / (iterator)


def gradient_word(data, W, T, i):
    crf_model = crf.crf(data[i][1], data[i][0], W, T)
    alpha, tmp, message = crf_model.forward()
    beta, tmp, message = crf_model.backward()

    w_grad = crf.w_grad(data[i][1], data[i][0], W, T)

    t_grad = crf.t_grad(data[i][1], data[i][0], W, T)

    # print(w_grad.shape, t_grad.shape)
    return np.concatenate((w_grad.flatten(), t_grad.flatten()))


def gradient_avg(params, data, iterator):
    total = np.zeros(128 * 26 + 26 * 26)
    W = compute.matricize_W(params)
    T = compute.matricize_Tij(params)

    for i in range(iterator):
        total += gradient_word(data, W, T, i)
    
    return total / (iterator)

def check_gradient(data, params):
    # print(gradient_avg(W, T, data, 1))
    grad_value = check_grad(compute_log_p_avg, gradient_avg, params, data, 10)
    print(grad_value)


if __name__ == "__main__":

    data = utils.read_data_seq('../data/train.txt')
    params = utils.load_model_params()
    # print(params.shape)

    # crf_model = crf.crf(X, Y, W, T)

    check_gradient(data, params)
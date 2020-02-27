import numpy as np

def compute_probability(x, y, W, T):
    # computes sum of <W_y, Xi> + sum of T_ij
    w_sum, t_sum = 0, 0

    for i in range(len(x)-1):
        w_sum += np.dot(x[i, :], W[y[i], :]) 
        t_sum += T[y[i], y[i+1]]
    n = len(x)-1
    w_sum += np.dot(x[n, :], W[y[n], :])

    return w_sum + t_sum

def comput_prob(x, y, W, T):
    w_x = np.dot(x, W.T)

    sum_val, t_sum = 0, 0

    for i in range(len(x)):
        sum_val += w_x[i][y[i]]

        if (i > 0):
            # t stored as T{current, prev}
            sum_val += T[y[i - 1]][y[i]]
    # print(sum_val)
    return np.exp(sum_val)








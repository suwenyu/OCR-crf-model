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
    w_sum, t_sum = 0, 0
    
    for i in range(len(x)-1):
        w_sum += np.dot(x[i], W[y[i]])
        t_sum += T[y[i], y[i+1]]

    n = len(x)-1
    w_sum += np.dot(x[n], W[y[n]])
    
    return w_sum + t_sum



def matricize_W(params):
    w = np.zeros((26, 128))

    for i in range(26):
        w[i] = params[128 * i: 128 * (i + 1)]

    return w


def matricize_Tij(params):
    t_ij = np.zeros((26, 26))

    index = 0
    for i in range(26):
        for j in range(26):
            t_ij[j][i] = params[128 * 26 + index]
            index += 1

    return t_ij

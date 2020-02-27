import compute
import numpy as np
import math

class crf:
    letter_size = 26

    def __init__(self, X, Y, W, T):
        self.X = X
        self.Y = Y
        self.W = W
        self.T = T
        self.n = len(X)

    # checked
    def forward(self):
        alpha = np.zeros((self.n, self.letter_size))
        # tmp = np.zeros(self.letter_size)
        message = np.zeros((self.letter_size, self.letter_size))

        w_x = np.dot(self.X, self.W.T)
        # equation (13)
        for i in range(1, self.n):
            tmp = alpha[i - 1] + self.T.transpose()
            tmp_max = np.max(tmp, axis=1)
            
            # print(tmp_max)
            tmp = (tmp.transpose() - tmp_max).transpose()

            tmp = np.exp(tmp + w_x[i - 1])
            # alpha_max = np.max(alpha, axis=1)
            # prepare V - V.max()
            # alpha = (alpha.transpose() - alpha_max).transpose()
            alpha[i] = tmp_max + np.log(np.sum(tmp, axis=1))
            # print(alpha[i])
            
            # for j in range(self.letter_size):
            #     for k in range(self.letter_size):
            #         message[k][j] = np.dot(self.W[k], self.X[i-1]) + self.T[k, j] + alpha[i-1, k]

            # not sure
            # maxes = message.max(axis=0)
            # message = np.add(message, -1*maxes)
            # 

            # message = np.exp(message)
            # tmp = np.sum(message, axis=0)
            # tmp = np.log(tmp)

            # not sure
            # alpha[i] = np.add(maxes, tmp)
            # 
            # alpha[i] = tmp
            # print(tmp)
        return alpha, tmp, message

    def backward(self):
        beta = np.zeros((self.n, self.letter_size))
        # tmp = np.zeros(self.letter_size)
        message = np.zeros((self.letter_size, self.letter_size))

        w_x = np.dot(self.X, self.W.T)
        # print(w_x)
        for i in range(self.n-2, -1, -1):
            tmp = beta[i + 1] + self.T

            tmp_max = np.max(tmp, axis=1)
            tmp = (tmp.transpose() - tmp_max).transpose()
            # print(tmp)
            tmp = np.exp(tmp + w_x[i + 1])
            beta[i] = tmp_max + np.log(np.sum(tmp, axis=1))

            # for j in range(self.letter_size):
            #     for k in range(self.letter_size):
            #         message[k][j] = np.dot(self.W[k], self.X[i+1]) + self.T[k, j] + beta[i+1, k]


            # np.swapaxes(message, 0, 1)
            # not sure
            # maxes = message.max(axis=0)
            # message = np.add(message, -1*maxes)
            # 

            # message = np.exp(message)
            # tmp = np.sum(message, axis=0)
            # tmp = np.log(tmp)
            # print(tmp)
            # not sure
            # beta[i] = np.add(maxes, tmp)
            # 
            # beta[i] = tmp
            # print(tmp)

        return beta, tmp, message

    def compute_z(self, alpha):
        # equation (14)
        # tmp = np.add(np.matmul(self.W, self.X[-1]), alpha[-1])
        # M = np.max(tmp)
        # log_z = M + math.log(np.sum(np.exp(np.add(tmp, -1*M))))

        # return log_z
        # print(np.sum(np.exp(alpha[-1] + np.dot(self.X, self.W.T)[-1])))
        # print(np.sum(np.exp(np.dot(self.X, self.W.T)[-1] )))
        return np.sum(np.exp(alpha[-1] + np.dot(self.X, self.W.T)[-1]))

    # checked
    def compute_log_prob(self):
        sum_num = compute.comput_prob(self.X, self.Y, self.W, self.T)
        
        alpha, tmp, message = self.forward()

        # print(sum_num, self.compute_z(alpha))
        # z = self.compute_z(alpha)
        # print(z)
        # print(sum_num-log_z)

        return np.log(sum_num / self.compute_z(alpha))


    def forward_backward_prob(self):

        alpha, tmp, message = self.forward()

        # log_z = self.cal_logz(tmp, alpha)

        beta, tmp, message = self.backward()
        # print(beta)
        return alpha, beta


def w_grad(X, Y, W, T, denom, alpha, beta):
    letter_size = 26
    n = len(X)

    # crf_model = crf(X, Y, W, T)
    # alpha, beta = crf_model.forward_backward_prob()

    grad = np.zeros((letter_size, 128))

    # tmp = np.ones((letter_size, 128))
    w_x = np.dot(X, W.T)
    # print(X)

    for i in range(n):
        # print(Y[i])
        grad[Y[i]] += X[i]
        # print(grad[Y[i]])
        # print(grad[Y[i]])
        
        # prob = np.add(alpha[i], beta[i])
        # prob = np.add(w_x[i], prob)
        # prob = np.exp(prob)


        tmp = np.ones((26, 128)) * X[i]
        tmp = tmp.transpose()
        # print(tmp)

        # print(denom)
        tmp = tmp * np.exp(alpha[i] + beta[i] + w_x[i]) / denom
        # print(tmp)
        # tmp = tmp * np.exp(prob) / denom
        # print(tmp.shape)
        grad -= tmp.transpose()
        # print(grad)
        # print(np.sum(grad))

        # expect[Y[i]] = 1
        # expect = np.add(expect, -1*prob)
        # letter_grad = np.tile(X[i], (26, 1))
        # letter_grad = np.multiply(expect[:, np.newaxis], letter_grad)
        # grad = np.add(grad, letter_grad)
        # expect[:] = 0

    # print(grad.flatten())
    # for i in grad:
        # print(i)
    # for i in grad.flatten():
    #     print(i)
    return grad.flatten()


def t_grad(X, Y, W, T, denom, alpha, beta):
    n = len(X)
    letter_size = 26
    
    grad = np.zeros((letter_size, letter_size))

    w_x = np.dot(X, W.T)

    # crf_model = crf(X, Y, W, T)

    # alpha, beta = crf_model.forward_backward_prob()

    for i in range(n - 1):
        for j in range(26):
            grad[j] -= np.exp(w_x[i] + T.transpose()[j] + w_x[i + 1][j] + beta[i + 1][j] + alpha[i])
            
            # print(grad[j])
            # tmp = np.exp(beta[i + 1][j] + alpha[i] + w_x[i + 1][j] + w_x[i] + T[j])
            # print(tmp)
            # grad[j] -= tmp

            # grad[j][j+1] = 
            # np.exp(w_x[i] + T.transpose()[j] + w_x[i + 1][j] )

    # print(grad)
    grad /= denom

    for i in range(n - 1):
        grad[Y[i+1]][Y[i]] += 1

    # for i in range(len(w_x) - 1):
    #     t_index = y[i]
    #     t_index += 26 * y[i + 1]
    #     # print(t_index, y[i], y[i+1])
    #     gradient[t_index] += 1

    # print(grad)
    # for i in grad.flatten():
    #     print(i)
    return grad



if __name__ == "__main__":
    import utils
    W, T = utils.read_model()
    data = utils.read_data_seq('../data/train.txt')

    # crf_model = crf(data[0][1], data[0][0], W, T)
    # crf_model.forward_backward_prob()
    # print(T.shape)

    # t_grad(data[0][1], data[0][0], W, T)




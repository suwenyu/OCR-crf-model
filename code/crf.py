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


    def forward(self):
        alpha = np.zeros((self.n, self.letter_size))
        # tmp = np.zeros(self.letter_size)
        message = np.zeros((self.letter_size, self.letter_size))

        # equation (13)
        for i in range(1, self.n):
            for j in range(self.letter_size):
                for k in range(self.letter_size):
                    message[k][j] = np.dot(self.W[k], self.X[i-1]) + self.T[k, j] + alpha[i-1, k]

            # not sure
            maxes = message.max(axis=0)
            message = np.add(message, -1*maxes)
            # 

            message = np.exp(message)
            tmp = np.sum(message, axis=0)
            tmp = np.log(tmp)

            # not sure
            alpha[i] = np.add(maxes, tmp)
            # 
            # alpha[i] = tmp
            # print(tmp)
        return alpha, tmp, message
        

    def backward(self):
        beta = np.zeros((self.n, self.letter_size))
        # tmp = np.zeros(self.letter_size)
        message = np.zeros((self.letter_size, self.letter_size))

        for i in range(self.n-2, -1, -1):
            for j in range(self.letter_size):
                for k in range(self.letter_size):
                    message[k][j] = np.dot(self.W[k], self.X[i+1]) + self.T[k, j] + beta[i+1, k]


            # np.swapaxes(message, 0, 1)
            # not sure
            maxes = message.max(axis=0)
            message = np.add(message, -1*maxes)
            # 

            message = np.exp(message)
            tmp = np.sum(message, axis=0)
            tmp = np.log(tmp)
            # print(tmp)
            # not sure
            beta[i] = np.add(maxes, tmp)
            # 
            # alpha[i] = tmp
            # print(tmp)


        return beta, tmp, message

    def cal_logz(self, tmp, alpha):
        # equation (14)
        tmp = np.add(np.matmul(self.W, self.X[-1]), alpha[-1])
        M = np.max(tmp)
        log_z = M + math.log(np.sum(np.exp(np.add(tmp, -1*M))))

        return log_z


    def log_prob(self):
        sum_num = compute.comput_prob(self.X, self.Y, self.W, self.T)
        
        alpha, tmp, message = self.forward()
        log_z = self.cal_logz(tmp, alpha)
        
        return sum_num - log_z


    def forward_backward_prob(self):

        alpha, tmp, message = self.forward()

        log_z = self.cal_logz(tmp, alpha)

        beta, tmp, message = self.backward()
        # print(beta)
        return alpha, beta, log_z


def w_grad(X, Y, W, T):
    letter_size = 26
    n = len(X)
    grad = np.zeros((letter_size, letter_size))

    crf_model = crf(X, Y, W, T)
    fw, bw, log_z = crf_model.forward_backward_prob()

    grad = np.zeros((letter_size, 128))
    expect = np.zeros(letter_size)
    prob = np.zeros(letter_size)

    for i in range(n):
        prob = np.add(fw[i, : ], bw[i, :])
        prob = np.add(np.matmul(W, X[i]), prob)
        prob = np.add(-1*log_z, prob)
        prob = np.exp(prob)

        expect[Y[i]] = 1
        expect = np.add(expect, -1*prob)
        letter_grad = np.tile(X[i], (26, 1))
        letter_grad = np.multiply(expect[:, np.newaxis], letter_grad)
        grad = np.add(grad, letter_grad)
        expect[:] = 0

    print(prob)
    return grad


if __name__ == "__main__":
    import utils
    W, T = utils.read_model()
    data = utils.read_data_seq('../data/train.txt')

    # crf_model = crf(data[0][1], data[0][0], W, T)
    # crf_model.forward_backward_prob()

    w_grad(data[0][1], data[0][0], W, T)




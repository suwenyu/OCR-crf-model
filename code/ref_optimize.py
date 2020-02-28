# Imports
import numpy as np
import scipy.optimize as opt

import crf, utils, compute, check_grad, train_crf
import time

def crf_obj(x, word_list, c):
    """Compute the CRF objective and gradient on the list of words (word_list)
    evaluated at the current model x (w_y and T, stored as a vector)
    """
    
    # x is a vector as required by the solver. So reshape it to w_y and T
    # W = np.reshape(x[128*26], (128, 26))  # each column of W is w_y (128 dim)
    # T = np.reshape(x[128*26:], (26, 26))  # T is 26*26

    # f = get_crf_obj(word_list, W, T, c)  # Compute the objective value of CRF
                                         # objective log-likelihood + regularizer

    f = train_crf.func(x, word_list, c)
    g = train_crf.func_prime(x, word_list, c)

    # g_W = blah                  # compute the gradient in W(128 * 26)
    # g_T = blah                  # compute the gradient in T(26*26)
    # g = np.concatenate([g_W.reshape(-1), g_T.reshape(-1)])  # Flatten the
                                                          # gradient back into
                                                          # a vector
    return [f,g]

def crf_test(x, word_list):
    """
    Compute the test accuracy on the list of words (word_list); x is the
    current model (w_y and T, stored as a vector)
    """

    # x is a vector. so reshape it into w_y and T
    # W = np.reshape(x[:128*26], (128, 26))  # each column of W is w_y (128 dim)
    # T = np.reshape(x[128*26:], (26, 26))  # T is 26*26

    W = utils.extract_w(x)
    T = utils.extract_t(x)

    # Compute the CRF prediction of test data using W and T
    y_predict = train_crf.decode_test_data(word_list, W, T)

    #get y_label data from test_data
    true_label_of_word_list = []
    for d in test_data:
        true_label_of_word_list.append(d[0])

    # Compute the test accuracy by comparing the prediction with the ground truth
    word_acc, letter_acc = train_crf.word_letter_accuracy(y_predict, true_label_of_word_list)
    print('Word Accuracy = {}, Letter Accuracy = {}.\n'.format(word_acc, letter_acc))
    return letter_acc

def ref_optimize(train_data, test_data, c, params):
    print('Training CRF ... c = {} \n'.format(c))

    # Initial value of the parameters W and T, stored in a vector
    # x0 = np.zeros((128*26+26**2,1))

    # Start the optimization
    result = opt.fmin_tnc(crf_obj, params, args = [train_data, c], maxfun=100,
                          ftol=1e-3, disp=5)
    model  = result[0]          # model is the solution returned by the optimizer

    accuracy = crf_test(model, test_data)
    print('CRF test accuracy for c = {}: {}'.format(c, accuracy))
    # return accuracy

def read_data():
    train_data = utils.read_data_seq('../data/train.txt')
    test_data = utils.read_data_seq('../data/test.txt')

    params = utils.load_model_params('../data/model.txt')
    # train_data = (trainX, trainY)
    # test_data = (testX, testY)

    return train_data, test_data, params

if __name__ == '__main__':
    print('Reading data ...')
    train_data, test_data, params = read_data()

    c = 1000
    ref_optimize(train_data, test_data, c, params)




    
    

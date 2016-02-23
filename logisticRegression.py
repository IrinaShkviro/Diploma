# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:19:49 2016

@author: irina
"""

__docformat__ = 'restructedtext en'

import numpy

import theano
import theano.tensor as T

from binaryReader import BinaryReader
from sgd import train_logistic_sgd
#from cg import train_logistic_cg
from base import errors
from visualizer import vis_log_reg

class LogisticRegression(object):
    def __init__(self, rng, input, n_in, n_out):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        """
        
        self.input = input
        self.n_in = n_in
        self.n_out = n_out

        # initialize theta = (W,b) with random values;
        # W gets the shape (n_in, n_out),
        # while b is a vector of n_out elements
        # making theta a vector of n_in*n_out + n_out elements
        theta_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=n_in * n_out + n_out
            ),
            dtype=theano.config.floatX
        )
        self.theta = theano.shared(
            value=theta_values,
            name='theta',
            borrow=True
        )        
        
        # separate W from theta
        self.W = self.theta[0:n_in * n_out].reshape((n_in, n_out))
        
        # separate b from theta
        self.b = self.theta[n_in * n_out:n_in * n_out + n_out]

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]
        
        self.valid_error_array = []
        self.train_cost_array = []
        self.train_error_array = []
        self.epoch = 0
        
    def print_log_reg_types(self):
        print(self.W.type(), 'W')
        print(self.b.type(), 'b')
        print(self.p_y_given_x.type(), 'p_y_given_x')
        print(self.y_pred.type(), 'y_pred')
        

    def negative_log_likelihood(self, y):
        """
        Return the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a number that gives for each example the
                  correct label
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1]
        # T.log(self.p_y_given_x) is a matrix of Log-Probabilities (call it LP)
        # with one row per example and one column per class
        # LP[T.arange(y.shape[0]),y] is a vector v containing [LP[0,y[0]],
        # LP[1,y[1]], LP[2,y[2]], ..., LP[n-1,y[n-1]]] 
        # T.mean(LP[T.arange(y.shape[0]),y]) is the mean (across minibatch examples)
        # of the elements in v, i.e., the mean log-likelihood across the minibatch.
        
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
            
    def predict(self):
        """
        Return predicted y
        """
        return self.y_pred
        
    def distribution(self):
        return self.p_y_given_x
        
    def errors(self, y):
        return errors(
            predicted = self.y_pred,
            actual = y,
            not_shared = False
        )
        
def train_log_reg(learning_rate,
                 n_epochs,
                 base_folder,
                 train_algo = 'sgd',
                 batch_size = 1):
    
    x = T.matrix('x')
    rng = numpy.random.RandomState(1234)
    classifier = LogisticRegression(
        rng=rng,
        input=x,
        n_in=75,
        n_out=39
    )
    
    if (train_algo == 'sgd'):
        trained_classifier = train_logistic_sgd(
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            classifier = classifier,
            batch_size = batch_size
        )
    else:
        trained_classifier = train_logistic_cg(
            n_epochs = n_epochs,
            classifier = classifier
        )
        
    vis_log_reg(
        base_folder = base_folder,
        train_cost = classifier.train_cost_array,
        train_error = classifier.train_error_array,
        valid_error = classifier.valid_error_array,
        learning_rate = learning_rate
    )
    
    return trained_classifier
    
def test_log_reg(classifier):
    
    test_reader = BinaryReader(isTrain=False)    
    y = T.ivector('y')
    
    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input, y],
        outputs=classifier.errors(y)
    )
    
    test_error_array = []    
    
    for pat_num in xrange(test_reader.n_files):
        test_set_x, test_set_y = test_reader.read_next_doc()        
        test_error_array.append(predict_model(
            test_set_x.get_value(borrow=True),
            test_set_y.eval()
        ))
     
    return test_error_array
        
def test_all_params():  
    learning_rate = 0.0001
    
    n_epochs = 2
    train_algo = 'sgd'
    
    trained_classifier = train_log_reg(
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        base_folder=('log_reg_%s')%(train_algo),
        train_algo = 'sgd',
        batch_size = 1
    )
    test_errors = test_log_reg(
        classifier = trained_classifier
    )
    print('errors:', test_errors)
    print('mean value of error: ', numpy.round(numpy.mean(test_errors), 6))
    print('min value of error: ', numpy.round(numpy.amin(test_errors), 6))
    print('max value of error: ', numpy.round(numpy.amax(test_errors), 6))    

if __name__ == '__main__':
    test_all_params()

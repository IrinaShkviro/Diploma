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
from sgd_with_stopping import train_logistic_sgd
#from cg import train_logistic_cg
from base import errors
from visualizer import vis_log_reg

class LogisticRegression(object):
    def __init__(self, rng, input, n_in, n_out):
        '''
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        '''
        
        self.input = input
        self.n_in = n_in
        self.n_out = n_out

        rng.seed()
        
        self.W = theano.shared(
            value = numpy.asarray(
                rng.uniform(
                    low=0,
                    high=0.05,
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.asarray(
                rng.uniform(
                    low=0,
                    high=0.05,
                    size=(n_out,)
                ),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        
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
        self.validation = numpy.inf
        self.best_cost = numpy.inf
        self.best_error = numpy.inf
        
    def negative_log_likelihood(self, y):
        '''
        Return the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a number that gives for each example the
                  correct label
        '''
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
        ''' Return predicted y '''
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
                 pat_epochs,
                 base_folder,
                 n_features,
                 n_classes,
                 train_algo = 'sgd',
                 batch_size = 1,
                 attempt = 0,
                 global_epochs = 1,
                 train_seq_len = 20,
                 test_seq_len = 40):
    
    x = T.vector('x')
    rng = numpy.random.RandomState()
    classifier = LogisticRegression(
        rng = rng,
        input = x,
        n_in = n_features,
        n_out = n_classes
    )
    
    if (train_algo == 'sgd'):
        trained_classifier = train_logistic_sgd(
            learning_rate = learning_rate,
            pat_epochs = pat_epochs,
            classifier = classifier,
            batch_size = batch_size,
            global_epochs = global_epochs,
            train_seq_len = train_seq_len,
            test_seq_len = test_seq_len
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
        learning_rate = learning_rate,
        attempt = attempt,
        batch_size = batch_size
    )
    
    return trained_classifier
    
def test_log_reg(classifier, test_seq_len):    
    test_reader = BinaryReader(
        isTrain=False,
        len_seqs = test_seq_len
    )    
    y = T.ivector('y')
    
    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input, y],
        outputs=classifier.errors(y)
    )
    
    test_error_array = []    
    
    for pat_num in xrange(test_reader.n_files):
        test_set_x, test_set_y = test_reader.read_several()        
        test_error_array.append(predict_model(
            test_set_x.get_value(
                borrow=True,
                return_internal_type=True
            ),
            test_set_y.eval()
        ))
     
    return test_error_array
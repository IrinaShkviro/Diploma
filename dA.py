# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:17:32 2016

@author: irina
"""

import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

#from MyVisualizer import visualize_da
#from sgd import train_da_sgd
#from cg import train_da_cg

class dA(object):
    def __init__(
        self,
        numpy_rng,
        n_visible,
        n_hidden,
        theano_rng=None,
        input=None,
        W=None,
        bvis=None,
        bhid=None
    ):
        """
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # initialize theta = (W,b) with 0s; W gets the shape (n_visible, n_hidden),
        # while b is a vector of n_out elements, making theta a vector of
        # n_visible*n_hidden + n_hidden elements
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=0,
                    high=0.05,
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)
        
        if not bvis:
            bvis_values = numpy.asarray(
                numpy_rng.uniform(
                    low=0,
                    high=0.05,
                    size=(n_visible,)
                ),
                dtype=theano.config.floatX
            )
            bvis = theano.shared(
                value=bvis_values,
                borrow=True
            )
            
        if not bhid:
            bhid = theano.shared(
                value=numpy_rng.uniform(
                    low=0,
                    high=0.05,
                    size=(n_hidden,)
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            self.x = T.matrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]
        
        self.train_cost_array=[]
        self.valid_error_array = []
        self.epoch=0
        self.best_cost = numpy.inf

    def get_corrupted_input(self, input, corruption_level):
        """ This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level`` """
        return self.theano_rng.binomial(
            size=input.shape,
            n=1,
            p=1 - corruption_level,
            dtype=theano.config.floatX
        ) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost(self, corruption_level):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        
        cost = T.sqrt(T.sum(T.sqr(T.flatten(self.x - z))))
                
        return cost
        
    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)
        
    def predict(self):
        """
        Return predicted vector
        """
        return self.z
    
    def actual(self):
        """
        Return actual vector
        """
        return self.x    
        
def train_dA(
    learning_rate,
    training_epochs,
    window_size,
    corruption_level,
    n_hidden,
    train_set,
    output_folder,
    train_algo="sgd"):

    """
    This dA is tested on ICHI_Data

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type window_size: int
    :param window_size: size of window used for training

    :type corruption_level: float
    :param corruption_level: corruption_level used for training the DeNosing
                          AutoEncoder

    :type n_hidden: int
    :param n_hidden: count of nodes in hidden layer

    :type output_folder: string
    :param output_folder: folder for costand error graphics with results

    """
    
    start_time = time.clock()
    
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    
    x = T.vector('x')    

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=window_size,
        n_hidden=n_hidden
    )
    
    if train_algo == "sgd":
        updated_da = train_da_sgd(
            learning_rate=learning_rate,
            window_size=window_size,
            training_epochs=training_epochs,
            corruption_level=corruption_level,
            train_set=train_set,
            da=da
        )
        base_folder = "da_sgd"
    else:
        updated_da = train_da_cg(
            da=da,
            train_set=train_set,
            window_size=window_size,
            corruption_level=corruption_level,
            training_epochs=training_epochs
        )
        base_folder = "da_cg"
    
    visualize_da(train_cost=updated_da.train_cost_array,
                 window_size=window_size,
                 learning_rate=learning_rate,
                 corruption_level=corruption_level,
                 n_hidden=n_hidden,
                 output_folder=output_folder,
                 base_folder=base_folder)
    
    end_time = time.clock()
    training_time = (end_time - start_time)
    
    print >> sys.stderr, ('The with corruption %f code for file ' +
                          os.path.split(__file__)[1] +
                         ' ran for %.2fm' % (corruption_level, (training_time) / 60.))

def test_da_params(corruption_level):
    learning_rates = [0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.013, 0.015]
    window_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    
    train_data = ['p10a','p011','p013','p014','p020','p022','p040','p045','p048']
    valid_data = ['p09b','p023','p035','p038']
    test_data = ['p09a','p033']
    
    train_reader = ICHISeqDataReader(train_data)
    train_set, train_labels = train_reader.read_all()
    
    valid_reader = ICHISeqDataReader(valid_data)
    valid_set, valid_labels = valid_reader.read_all()

    test_reader = ICHISeqDataReader(test_data)
    test_set, test_labels = test_reader.read_all()
    
    output_folder=('[%s], [%s], [%s]')%(",".join(train_data), ",".join(valid_data), ",".join(test_data))
    
    for lr in learning_rates:
        for ws in window_sizes:
            train_dA(learning_rate=lr,
                     training_epochs=1,
                     window_size = ws, 
                     corruption_level=corruption_level,
                     n_hidden=ws*2,
                     train_set=train_set,
                     output_folder=output_folder)


if __name__ == '__main__':
    test_da_params(corruption_level=0.)
    test_da_params(corruption_level=0.3)

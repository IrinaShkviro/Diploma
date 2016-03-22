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
        W,
        bhid,
        input,
        theano_rng=None,
        bvis=None
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
        
        if not bvis:
            bvis_values = numpy.asarray(
                numpy_rng.uniform(
                    low=0,
                    high=0.05,
                    size=(1, n_visible)
                ),
                dtype=theano.config.floatX
            )
            bvis = theano.shared(
                value=bvis_values,
                borrow=True,
                broadcastable=(True, False)
            )
            
        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        
        self.x = input.reshape((-1, n_visible))

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

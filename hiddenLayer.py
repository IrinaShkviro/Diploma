# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:18:38 2016

@author: irina
"""

__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input.reshape((-1, n_in))
        rng.seed()
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=0,
                    high=0.05,
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)
            
        if b is None:
            b_values = numpy.zeros((1,n_out), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True, broadcastable=(True, False))

        self.W = W
        self.b = b
        
        
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        
        self.params = [self.W, self.b]
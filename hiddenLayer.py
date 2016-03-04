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
    def __init__(self, rng, input, n_in, n_out, theta=None, activation=T.tanh):
        self.input = input
        rng.seed()
        if theta is None:
            theta_values = numpy.asarray(
                rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_out + n_in + 1)),
                    high=4 * numpy.sqrt(6. / (n_out + n_in + 1)),
                    size=(n_in * n_out + n_out)
                ),
                dtype=theano.config.floatX
            )
            theta = theano.shared(
                value=theta_values,
                name='theta',
                borrow=True
        )
        self.theta = theta
        
        # W is represented by the fisr n_visible*n_hidden elements of theta
        self.W = self.theta[0:n_in * n_out].reshape((n_in, n_out))
        # b is the rest (last n_hidden elements)
        self.b = self.theta[n_in * n_out:n_in * n_out + n_out]
        
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:21:59 2016

@author: irina
"""

import gc

import numpy

import theano
import theano.tensor as T

from binaryReader import BinaryReader
    
def training_functions_log_reg_sgd(classifier, batch_size=1):
    ''' Generates a list of functions, each of them implementing one
    step in trainnig the dA corresponding to the layer with same index.
    The function will require as input the minibatch index, and to train
    a dA you just need to iterate, calling the corresponding function on
    all minibatch indexes.

    :type train_set_x: theano.tensor.TensorType
    :param train_set_x: Shared variable that contains all datapoints used
                        for training the dA

    :type window_size: int
    :param window_size: size of a window

    :type learning_rate: float
    :param learning_rate: learning rate used during training for any of
                              the dA layers
    '''

    # allocate symbolic variables for the data
    index = T.lscalar('index')

    # generate symbolic variables for input
    y = T.ivector('y')
    cost = classifier.negative_log_likelihood(y)

    train_set_x = T.matrix('train_set_x')
    train_set_y = T.ivector('train_set_y')    
    learning_rate = T.scalar('lr')  # learning rate to use
    
    # compute the gradient of cost with respect to theta = (W,b)
    g_theta = T.grad(cost=cost, wrt=classifier.theta)
    
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.theta, classifier.theta - learning_rate * g_theta)]
    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    # cost and errors are mean values so they are scalars
    # predict and y are vectors for current minibatch
    train_model = theano.function(
        inputs=[
            index,
            train_set_x,
            train_set_y,
            theano.Param(learning_rate, default=0.001)
        ],
        outputs=[cost, classifier.errors(y), classifier.predict(), y],
        updates=updates,
        givens={
            classifier.input: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        inputs=[classifier.input, y],
        outputs=[classifier.errors(y)]
    )

    return train_model, validate_model

def train_logistic_sgd(
        learning_rate,
        n_epochs,
        classifier,
        batch_size=1
    ):
                          
    # read the datasets
    train_reader = BinaryReader(isTrain=True)
    
        
    # early-stopping parameters    
    patience_increase = 25  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    best_validation_loss = numpy.inf

    done_looping = False
    iter = 0
    classifier.train_cost_array = []
    classifier.train_error_array = []
    classifier.valid_error_array = []
    
    train_model, validate_model = training_functions_log_reg_sgd(
        classifier = classifier,
        batch_size = batch_size
    )
        
    for pat_num in xrange (train_reader.n_files):
        pat_epoch = 0
        # go through the training set
        train_set_x, train_set_y = train_reader.read_next_doc()        
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        
        patience = 5000  # look as this many examples regardless
        validation_frequency = min(n_train_batches, patience // 2)
        
        done_looping = False
        
        while (pat_epoch < n_epochs) and (not done_looping):
            cur_train_cost =[]
            cur_train_error = []
            for index in xrange(n_train_batches):            
                mean_cost, mean_error, cur_pred, cur_actual = train_model(
                    index = index,
                    train_set_x = train_set_x.get_value(borrow=True),
                    train_set_y = train_set_y.eval(),
                    lr = learning_rate
                )
                # iteration number
                iter = pat_epoch * n_train_batches + index
                    
                cur_train_cost.append(mean_cost)
                cur_train_error.append(mean_error)
            
                if (iter + 1) % validation_frequency == 0:
                    valid_reader = BinaryReader(isTrain=False)
                    # compute zero-one loss on validation set
                    valid_error_array = []    
    
                    for pat_num in xrange(valid_reader.n_files):
                        valid_features, valid_labels = valid_reader.read_next_doc()        
                        valid_error_array.append(validate_model(
                            valid_features.get_value(borrow=True),
                            valid_labels.eval()
                        ))
        
                    this_validation_loss = float(numpy.mean(valid_error_array))*100                 
                    classifier.valid_error_array.append([])
                    classifier.valid_error_array[-1].append(classifier.epoch + float(index)/n_train_batches)
                    classifier.valid_error_array[-1].append(this_validation_loss)
           
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                            improvement_threshold:
                            patience = max(patience, iter * patience_increase)
            
                        best_validation_loss = this_validation_loss

                if patience <= iter:
                    done_looping = True
                    break
                               
            classifier.epoch = classifier.epoch + 1
            pat_epoch = pat_epoch + 1
            
            classifier.train_cost_array.append([])
            classifier.train_cost_array[-1].append(classifier.epoch)
            classifier.train_cost_array[-1].append(float(numpy.mean(cur_train_cost)))
            cur_train_cost =[]
           
            classifier.train_error_array.append([])
            classifier.train_error_array[-1].append(classifier.epoch)
            classifier.train_error_array[-1].append(float(numpy.mean(cur_train_error)*100))
            cur_train_error =[]
                    
            gc.collect()
                        
    return classifier
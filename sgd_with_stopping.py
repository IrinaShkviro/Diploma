# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 16:08:00 2016

@author: irina
"""

import gc
import os

import numpy
import pickle

import theano
import theano.tensor as T

from binaryReader import BinaryReader
    
def training_functions_log_reg_sgd(classifier, batch_size=1):
    ''' 
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
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]
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
        outputs=[cost, classifier.errors(y)],
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
        pat_epochs,
        classifier,
        batch_size=1,
        global_epochs=1,
        train_seq_len=20,
        test_seq_len=40
    ):
                          
    # read the datasets
    train_reader = BinaryReader(
        isTrain=True,
        len_seqs=train_seq_len
    )
    best_validation_loss = numpy.inf

    train_model, validate_model = training_functions_log_reg_sgd(
        classifier = classifier,
        batch_size = batch_size
    )
    
    n_failures = 0
    done_global_loop = False
    max_failures = train_reader.n_files
    global_index = 0    
    
    while (global_index < global_epochs) and (not done_global_loop):
        global_index = global_index + 1
        pat_num = 0
        while (pat_num < train_reader.n_files) and (not done_global_loop):
            pat_num = pat_num + 1
            done_looping = False
            pat_epoch = 0
            # go through the training set
            train_features, train_labels = train_reader.read_several()
            train_features = train_features.get_value(
                borrow=True,
                return_internal_type=True
            )
            train_labels = train_labels.eval()
            n_train_batches = train_features.shape[0] // batch_size
            
            iter = 0
            # early-stopping parameters
            improvement_threshold = 0.995  # a relative improvement of this much is
                                           # considered significant
            n_local_failures = 0
            max_local_failures = 3
            
            validation_frequency = n_train_batches // max_local_failures
            validation_increase = 0.2

            while (pat_epoch < pat_epochs) and (not done_looping):
                classifier.epoch = classifier.epoch + 1
                pat_epoch = pat_epoch + 1
                
                cur_train_cost =[]
                cur_train_error = []
                for minibatch_index in xrange(n_train_batches):
                    mean_cost, mean_error = train_model(
                        index = minibatch_index,
                        train_set_x = train_features,
                        train_set_y = train_labels,
                        lr = learning_rate
                    )
                    # iteration number: batch's number that we train now
                    iter = iter + 1
                        
                    cur_train_cost.append(mean_cost)
                    cur_train_error.append(mean_error)
                
                    if iter % validation_frequency == 0:
                        valid_reader = BinaryReader(
                            isTrain=False,
                            len_seqs=test_seq_len
                        )
                        # compute zero-one loss on validation set
                        valid_error_array = []    
        
                        for seq_index in xrange(valid_reader.n_files):
                            valid_features, valid_labels = valid_reader.read_several()        
                            valid_error_array.append(validate_model(
                                valid_features.get_value(
                                    borrow=True,
                                    return_internal_type=True
                                ),
                                valid_labels.eval()
                            ))
            
                        this_validation_loss = float(numpy.mean(valid_error_array))*100                 
                        classifier.valid_error_array.append([])
                        classifier.valid_error_array[-1].append(classifier.epoch + float(minibatch_index)/n_train_batches)
                        classifier.valid_error_array[-1].append(this_validation_loss)
               
                        # if we got the best validation score until now
                        if this_validation_loss < best_validation_loss:
                              best_validation_loss = this_validation_loss
                              n_failures = 0
                              n_local_failures = 0                              
                              
                              #improve patience if loss improvement is good enough
                              if this_validation_loss < best_validation_loss *  \
                              improvement_threshold:
                                  validation_frequency = int(validation_frequency * \
                                      validation_increase)
                                  max_local_failures = n_train_batches // \
                                      validation_frequency
                        else:
                            n_local_failures = n_local_failures + 1
                            if n_local_failures > max_local_failures:
                                done_looping = True
                                n_failures = n_failures + 1
                                if n_failures > max_failures:
                                    done_global_loop = True
                                break
                        
                        gc.collect()
                                                                   
                classifier.train_cost_array.append([])
                classifier.train_cost_array[-1].append(float(classifier.epoch))
                classifier.train_cost_array[-1].append(float(numpy.mean(cur_train_cost)))
                cur_train_cost =[]
               
                classifier.train_error_array.append([])
                classifier.train_error_array[-1].append(float(classifier.epoch))
                classifier.train_error_array[-1].append(float(numpy.mean(cur_train_error)*100))
                cur_train_error =[]
                
            gc.collect()
        
    valid_reader = BinaryReader(
        isTrain=False,
        len_seqs=test_seq_len
    )
    valid_error_array = []
    
    for pat_num in xrange(valid_reader.n_files):
        valid_features, valid_labels = valid_reader.read_several()        
        valid_error_array.append(validate_model(
            valid_features.get_value(
                borrow=True,
                return_internal_type=True
            ),
            valid_labels.eval()
        ))
        
    this_validation_loss = float(numpy.mean(valid_error_array))*100                 
    classifier.valid_error_array.append([])
    classifier.valid_error_array[-1].append(float(classifier.epoch))
    classifier.valid_error_array[-1].append(this_validation_loss)                   
    classifier.validation = this_validation_loss
    return classifier
    
def pretraining_functions(sda, batch_size):
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

    index = T.lscalar('index')
    corruption_level = T.scalar('corruption')  # % of corruption to use
    learning_rate = T.scalar('lr')  # learning rate to use
    train_set = T.matrix('train_set')

    pretrain_fns = []
    for cur_dA in sda.dA_layers:
        # get the cost and the updates list
        cost, updates = cur_dA.get_cost_updates(
            corruption_level=corruption_level,
            learning_rate=learning_rate
        )

        # compile the theano function
        fn = theano.function(
            inputs=[
                index,
                train_set,
                theano.Param(corruption_level, default=0.2),
                theano.Param(learning_rate, default=0.1)
            ],
            outputs=cost,
            updates=updates,
            givens={
                sda.x: train_set[index * batch_size: index * batch_size + batch_size]
            }
        )
        
        # append `fn` to the list of functions
        pretrain_fns.append(fn)
        
    return pretrain_fns
    
def pretrain_many_sda_sgd(
        sda,
        pretrain_lr,
        corruption_levels,
        global_epochs,
        pat_epochs,
        batch_size,
        debug_folder,
        debug_file,
        train_seq_len=20,
        test_seq_len=40,
        n_attempts=1):
            
    os.chdir(debug_folder)
    f = open(debug_file, 'a')
    f.write('\n START PRETRAINING \n')
    f.close()
    os.chdir('../')    
        
    init_folder = 'init_das'
    if not os.path.isdir(init_folder):
        os.makedirs(init_folder)
    if not os.path.isdir('best_models'):
        os.makedirs('best_models')
        
    pretraining_fns = pretraining_functions(
        sda=sda,
        batch_size=batch_size
    )
    ## Pre-train layer-wise
    for i in xrange(sda.n_layers):
        os.chdir(debug_folder)
        f = open(debug_file, 'a')
        f.write('\npretrain_layer: %i\n' % i)
        f.close()
        os.chdir('../')
        
        cur_dA = sda.dA_layers[i]
        cur_dA.train_cost_array = []
        best_attempt_cost = numpy.inf
        best_model_name = ('best_da_num_%i.pkl')%(i)
        
        # save clean model
        os.chdir(init_folder)
        init_da_name = ('init_num_%i.pkl')%(i)
        with open(init_da_name, 'wb') as f:
            pickle.dump(cur_dA, f)
        os.chdir('../')
        
        # it is also the best model
        os.chdir('best_models')
        init_da_name = ('init_num_%i.pkl')%(i)
        with open(best_model_name, 'wb') as f:
            pickle.dump(cur_dA, f)
        os.chdir('../')

        for cur_attempt in xrange(n_attempts):
            train_reader = BinaryReader(
                isTrain=True,
                len_seqs=train_seq_len
            )  
            iter = 0
            attempt_cost = []
            best_cost = numpy.inf
            
            # open clean model
            os.chdir(init_folder)
            cur_dA = pickle.load(open(init_da_name))
            os.chdir('../')
            sda.dA_layers[i] = cur_dA
            
            n_failures = 0
            done_global_loop = False
            max_failures = train_reader.n_files
            global_index = 0    
            
            while (global_index < global_epochs) and (not done_global_loop):
                global_index = global_index + 1
                pat_num = 0
                while (pat_num < train_reader.n_files) and (not done_global_loop):
                    pat_num = pat_num + 1
                    done_looping = False
                    # go through the training set
                    train_features, train_labels = train_reader.read_several()
                    train_features = train_features.get_value(
                        borrow=True,
                        return_internal_type=True
                    )
                    n_train_batches = train_features.shape[0] // batch_size
                    
                    iter = 0
                    # early-stopping parameters
                    improvement_threshold = 0.995  # a relative improvement of this much is
                                                   # considered significant
                    n_local_failures = 0
                    max_local_failures = 3
                    
                    validation_frequency = n_train_batches // max_local_failures
                    validation_increase = 0.2
                    
                    cur_epoch_cost = []
                    pat_epoch=0
                    # go through pretraining epochs
                    while (pat_epoch < pat_epochs) and (not done_looping):
                        pat_epoch = pat_epoch + 1

                        cur_dA.epoch = cur_dA.epoch + 1
                        cur_epoch_cost=[]                      
                        for index in xrange(n_train_batches):
                            # iteration number
                            iter = iter + 1
                            cur_cost = pretraining_fns[i](
                                index=index,
                                train_set = train_features,
                                corruption=corruption_levels[i],
                                lr=pretrain_lr
                            )
                            cur_epoch_cost.append(cur_cost)
                            
                            if iter % validation_frequency == 0:
                                # if we got the best validation score until now
                                if cur_cost < best_cost:
                                      best_cost = cur_cost
                                      n_failures = 0
                                      n_local_failures = 0
                                      
                                      #improve patience if loss improvement is good enough
                                      if cur_cost < best_cost * improvement_threshold:
                                          validation_frequency = int(validation_frequency * \
                                              validation_increase)
                                          max_local_failures = n_train_batches // \
                                              validation_frequency
                                else:
                                    n_local_failures = n_local_failures + 1
                                    if n_local_failures > max_local_failures:
                                        done_looping = True
                                        n_failures = n_failures + 1
                                        if n_failures > max_failures:
                                            done_global_loop = True
                                        break
                            
                        mean_cost = numpy.mean(cur_epoch_cost)                                                       
                        cur_dA.train_cost_array.append([])
                        cur_dA.train_cost_array[-1].append(float(cur_dA.epoch))
                        cur_dA.train_cost_array[-1].append(mean_cost)
                        
                    attempt_cost = numpy.concatenate((attempt_cost, cur_epoch_cost))
                        
                gc.collect()
            
            mean_attempt_cost = numpy.mean(attempt_cost)         
            os.chdir(debug_folder)
            f = open(debug_file, 'a')
            f.write('cur_attempt %i, ' % cur_attempt)
            f.write('mean_attempt_cost %f\n' % mean_attempt_cost)
            f.close()
            os.chdir('../')
            
            if best_attempt_cost > mean_attempt_cost:
                best_attempt_cost = mean_attempt_cost
                cur_dA.best_cost = best_attempt_cost
                
                #save the best model for cur_da
                os.chdir('best_models')
                with open(best_model_name, 'wb') as f:
                    pickle.dump(cur_dA, f)
                os.chdir('../')
                
        # load the best model in sda
        os.chdir('best_models')
        sda.dA_layers[i] = pickle.load(open(best_model_name))
        os.chdir('../')            
    return sda
    
def build_finetune_functions(sda, batch_size, learning_rate):

    index = T.lscalar('index')  # index to a [mini]batch
    train_set_x = T.matrix('train_set_x')
    train_set_y = T.ivector('train_set_y')    

    # compute the gradients with respect to the model parameters
    gparams = T.grad(sda.finetune_cost, sda.params)

    # compute list of fine-tuning updates
    updates = [
        (param, param - gparam * learning_rate)
        for param, gparam in zip(sda.params, gparams)
    ]

    train_fn = theano.function(
        inputs=[
            index,
            train_set_x,
            train_set_y
        ],
        outputs=sda.finetune_cost,
        updates=updates,
        givens={
            sda.x: train_set_x[index * batch_size: (index + 1) * batch_size],
            sda.y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_score = theano.function(
        inputs=[sda.x, sda.y],
        outputs=[sda.errors]
    )

    return train_fn, test_score
    
def finetune_log_layer_sgd(
    sda,
    batch_size,
    finetune_lr,
    global_epochs,
    pat_epochs,
    train_seq_len,
    test_seq_len):
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing functions for the model
    train_fn, validate_model = build_finetune_functions(
        sda=sda,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )
    
    iter = 0
    validation_frequency = 5000*pat_epochs*global_epochs

    for global_epoch in xrange(global_epochs):
        train_reader = BinaryReader(
            isTrain=True,
            len_seqs=train_seq_len
        )  
        for pat_num in xrange(train_reader.n_files):
            # go through the training set
            train_features, train_labels = train_reader.read_several()
            train_features = train_features.get_value(
                borrow=True,
                return_internal_type=True
            )
            train_labels = train_labels.eval()
            n_train_batches = train_features.shape[0] // batch_size
            
            pat_epoch = 0    
            while (pat_epoch < pat_epochs):
                pat_epoch = pat_epoch + 1
                sda.logLayer.epoch = sda.logLayer.epoch + 1
                cur_train_cost = []
                cur_train_error = []
                for batch_index in xrange(n_train_batches):          
                    sample_cost, sample_error, cur_pred, cur_actual = train_fn(
                        index = batch_index,
                        train_set_x = train_features,
                        train_set_y = train_labels
                    )
                    
                    # iteration number
                    iter = iter + 1
                        
                    cur_train_cost.append(sample_cost)
                    cur_train_error.append(sample_error)
        
                    if (iter + 1) % validation_frequency == 0:
                        valid_reader = BinaryReader(
                            isTrain=False,
                            len_seqs=test_seq_len
                        )
                        # compute zero-one loss on validation set
                        valid_error_array = [] 
                        for valid_pat in xrange(valid_reader.n_files):
                            valid_features, valid_labels = valid_reader.read_several()
                            valid_error_array.append(validate_model(
                                valid_features.get_value(
                                    borrow=True,
                                    return_internal_type=True
                                ),
                                valid_labels.eval()
                            ))
                        valid_mean_error = numpy.mean(valid_error_array)                        
                        sda.logLayer.valid_error_array.append([])
                        sda.logLayer.valid_error_array[-1].append(
                            sda.logLayer.epoch + float(batch_index)/n_train_batches
                        )
                        sda.logLayer.valid_error_array[-1].append(valid_mean_error)
                        
                        gc.collect()
                                                          
                sda.logLayer.train_cost_array.append([])
                sda.logLayer.train_cost_array[-1].append(float(sda.logLayer.epoch))
                sda.logLayer.train_cost_array[-1].append(numpy.mean(cur_train_cost))
               
                sda.logLayer.train_error_array.append([])
                sda.logLayer.train_error_array[-1].append(float(sda.logLayer.epoch))
                sda.logLayer.train_error_array[-1].append(numpy.mean(cur_train_error)*100)
                        
            gc.collect()
    
    sda.logLayer.epoch = sda.logLayer.epoch + 1
    valid_reader = BinaryReader(
        isTrain=False,
        len_seqs=test_seq_len
    )
    # compute zero-one loss on validation set
    valid_error_array = [] 
    for valid_pat in xrange(valid_reader.n_files):
        valid_features, valid_labels = valid_reader.read_several()
        valid_error_array.append(validate_model(
            valid_features.get_value(
                borrow=True,
                return_internal_type=True
            ),
            valid_labels.eval()
        ))
    valid_mean_error = numpy.mean(valid_error_array)                        
    sda.logLayer.valid_error_array.append([])
    sda.logLayer.valid_error_array[-1].append(float(sda.logLayer.epoch))
    sda.logLayer.valid_error_array[-1].append(valid_mean_error)
    sda.logLayer.validation = valid_mean_error
    
    gc.collect()
    return sda
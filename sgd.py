# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:21:59 2016

@author: irina
"""

import gc

import numpy

import theano
import theano.tensor as T
import pickle
import os

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

    iter = 0    
    train_model, validate_model = training_functions_log_reg_sgd(
        classifier = classifier,
        batch_size = batch_size
    )
      
    validation_frequency = 5000 * pat_epochs * global_epochs
    
    for global_index in xrange(global_epochs):
        for pat_num in xrange (train_reader.n_files):
            pat_epoch = 0
            # go through the training set
            train_features, train_labels = train_reader.read_several()
            train_features = train_features.get_value(
                borrow=True,
                return_internal_type=True
            )
            train_labels = train_labels.eval()
            n_train_batches = train_features.shape[0] // batch_size
            
            while (pat_epoch < pat_epochs):
                cur_train_cost =[]
                cur_train_error = []
                for index in xrange(n_train_batches):            
                    mean_cost, mean_error = train_model(
                        index = index,
                        train_set_x = train_features,
                        train_set_y = train_labels,
                        lr = learning_rate
                    )
                    # iteration number
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
                        classifier.valid_error_array[-1].append(classifier.epoch + float(index)/n_train_batches)
                        classifier.valid_error_array[-1].append(this_validation_loss)
               
                        # if we got the best validation score until now
                        if this_validation_loss < best_validation_loss:
                              best_validation_loss = this_validation_loss
                        
                        gc.collect()
                                   
                classifier.epoch = classifier.epoch + 1
                pat_epoch = pat_epoch + 1
                
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
    
def pretraining_functions(sda, train_set, batch_size):
    # index to a [mini]batch
    index = T.lscalar('index')  # index to a minibatch
    corruption_level = T.scalar('corruption')  # % of corruption to use
    learning_rate = T.scalar('lr')  # learning rate to use
    
    pretrain_fns = []
    for dA in sda.dA_layers:
        # get the cost and the updates list
        cost, updates = dA.get_cost_updates(
            corruption_level=corruption_level,
            learning_rate=learning_rate
        )
        # compile the theano function
        fn = theano.function(
            inputs=[
                index,
                theano.Param(corruption_level, default=0.2),
                theano.Param(learning_rate, default=0.01)
            ],
            outputs=cost,
            updates=updates,
            givens=[(sda.x, train_set[index * batch_size: index * batch_size+ batch_size])],
            on_unused_input='ignore'
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
    ## Pre-train layer-wise
    pretrain_fns_matrix = []
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
            iter = 0
            attempt_cost = []
            
            # open clean model
            os.chdir(init_folder)
            cur_dA = pickle.load(open(init_da_name))
            os.chdir('../')
            
            for global_epoch in xrange(global_epochs):
                train_reader = BinaryReader(
                    isTrain=True,
                    len_seqs=train_seq_len
                )
                for seq_num in xrange(train_reader.n_files):
                    # go through the training set
                    train_features, train_labels = train_reader.read_several()
                    if (i == 0):
                        pretraining_fns = pretraining_functions(sda=sda,
                                                            train_set = train_features,
                                                            batch_size=batch_size)
                        pretrain_fns_matrix.append(pretraining_fns)
                    train_features = train_features.get_value(
                        borrow=True,
                        return_internal_type=True
                    )
                    n_train_batches = train_features.shape[0] // batch_size
                    
                    cur_epoch_cost = []
                    # go through pretraining epochs
                    for pat_epoch in xrange(pat_epochs):
                        cur_dA.epoch = cur_dA.epoch + 1
                        cur_epoch_cost=[]                      
                        for index in xrange(n_train_batches):
                            # iteration number
                            iter = iter + 1
                        
                            cur_epoch_cost.append(
                                pretrain_fns_matrix[seq_num][i](
                                    index=index,
                                    corruption=corruption_levels[i],
                                    lr=pretrain_lr
                                )
                            )                            
                        mean_cost = numpy.mean(cur_epoch_cost)
                        cur_dA.train_cost_array.append([])
                        cur_dA.train_cost_array[-1].append(float(cur_dA.epoch))
                        cur_dA.train_cost_array[-1].append(mean_cost)
                    if global_epoch == global_epochs-1:    
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
    
def build_finetune_train(sda, dataset, batch_size, learning_rate):

    (train_set_x, train_set_y) = dataset
    index = T.lscalar('index')  # index to a [mini]batch

    # compute the gradients with respect to the model parameters
    gparams = T.grad(sda.finetune_cost, sda.params)

    # compute list of fine-tuning updates
    updates = [
        (param, param - gparam * learning_rate)
        for param, gparam in zip(sda.params, gparams)
    ]

    train_fn = theano.function(
        inputs=[index],
        outputs=[sda.finetune_cost, sda.errors],
        updates=updates,
        givens={
            sda.x: train_set_x[index * batch_size: (index + 1) * batch_size],
            sda.y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        name='train'
    )

    return train_fn
        
def build_finetune_valid(sda, dataset, batch_size):

    (valid_set_x, valid_set_y) = dataset

    # compute number of minibatches for training, validation and testing
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_valid_batches //= batch_size

    index = T.lscalar('index')  # index to a [mini]batch

    valid_score_i = theano.function(
        [index],
        sda.errors,
        givens={
            sda.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            sda.y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        },
        name='valid'
    )

    # Create a function that scans the entire validation set
    def valid_score():
        return [valid_score_i(i) for i in range(n_valid_batches)]

    return valid_score
    
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
    
    iter = 0
    validation_frequency = 500*pat_epochs*global_epochs

    for global_epoch in xrange(global_epochs):
        train_reader = BinaryReader(
            isTrain=True,
            len_seqs=train_seq_len
        )
        for pat_num in xrange(train_reader.n_files):
            # go through the training set
            train_features, train_labels = train_reader.read_several()
            n_train_batches = train_features.get_value(
                borrow=True,
                return_internal_type=True
            ).shape[0] // batch_size
            
            train_fn = build_finetune_train(
                sda=sda,
                dataset=(train_features, train_labels),
                batch_size=batch_size,
                learning_rate=finetune_lr
            )
            
            pat_epoch = 0    
            while (pat_epoch < pat_epochs):
                pat_epoch = pat_epoch + 1
                sda.logLayer.epoch = sda.logLayer.epoch + 1
                cur_train_cost = []
                cur_train_error = []
                for batch_index in xrange(n_train_batches):          
                    sample_cost, sample_error = train_fn(
                        index = batch_index
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
                            validate_model = build_finetune_valid(
                                sda=sda,
                                dataset=(valid_features, valid_labels),
                                batch_size=batch_size
                            )
                            valid_error_array.append(numpy.mean(validate_model()))
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
        validate_model = build_finetune_valid(
            sda=sda,
            dataset=(valid_features, valid_labels),
            batch_size=batch_size
        )
        valid_error_array.append(numpy.mean(validate_model()))
    valid_mean_error = numpy.mean(valid_error_array)                        
    sda.logLayer.valid_error_array.append([])
    sda.logLayer.valid_error_array[-1].append(float(sda.logLayer.epoch))
    sda.logLayer.valid_error_array[-1].append(valid_mean_error)
    sda.logLayer.validation = valid_mean_error
    
    gc.collect()
    return sda
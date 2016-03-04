# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 18:45:48 2016

@author: irina
"""

#set config
'''THEANO_FLAGS='floatX=float32, device=gpu?, force_device=False, warn_float64=warn,\
    allow_gc=True, openmp=True, cast_policy=numpy+floatX, int_division=floatX,\
    mode=FAST_RUN, profile=False, nvcc.fastmath=True, numpy.seterr_over=warn,\
    numpy.seterr_under=warn, numpy.seterr_invalid=warn, exception_verbosity=low,\
    cmodule.compilation_warning=True, cmodule.preload_cache=True'
'''
import numpy
import pickle

from logisticRegression import train_log_reg, test_log_reg
from sdA import train_sda, test_sda

#define some shared constants
n_features = 75
n_classes = 39

def train_one_classifier():
    learning_rate = 0.0001    
    pat_epochs = 1
    batch_size = 10
    train_algo = 'sgd'
    global_epochs = 1
    train_seq_len = 20
    test_seq_len = 40
    
    # 1st approach
    # long training for one classifier
    trained_classifier = train_log_reg(
        learning_rate = learning_rate,
        pat_epochs = pat_epochs,
        base_folder = ('log_reg_%s')%(train_algo),
        n_features = n_features,
        n_classes = n_classes,
        train_algo = 'sgd',
        batch_size = batch_size,
        global_epochs = global_epochs,
        train_seq_len = train_seq_len,
        test_seq_len = test_seq_len
    )
                     
    test_errors = test_log_reg(
        classifier = trained_classifier,
        test_seq_len = test_seq_len
    )
    print('mean value of error: ', numpy.round(numpy.mean(test_errors), 6))
    print('min value of error: ', numpy.round(numpy.amin(test_errors), 6))
    print('max value of error: ', numpy.round(numpy.amax(test_errors), 6))
    
def train_many_regs():
    learning_rate = 0.01
    pat_epochs = 20
    train_algo = 'sgd'
    batch_size = 128
    n_attempts = 10
    global_epochs = 1
    train_seq_len = 128
    test_seq_len = 128
    best_validation = numpy.inf
    
    # 2nd approach
    # short training with few classifiers, save the best model
    for attempt in xrange(n_attempts):
        trained_classifier = train_log_reg(
            learning_rate = learning_rate,
            pat_epochs = pat_epochs,
            base_folder = ('few_log_regs_%s')%(train_algo),
            n_features = n_features,
            n_classes = n_classes,
            train_algo = train_algo,
            batch_size = batch_size,
            attempt = attempt,
            global_epochs = global_epochs,
            train_seq_len = train_seq_len,
            test_seq_len = test_seq_len
        )
           
        print('attempt: ', attempt)
        print('validation value: ', trained_classifier.validation)
        if trained_classifier.validation < best_validation:
            best_validation = trained_classifier.validation
            print('best_attempt', attempt)
            # save the best model
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(trained_classifier, f)
            
    best_classifier = pickle.load(open('best_model.pkl'))
    best_test_errors = test_log_reg(
        classifier = best_classifier,
        test_seq_len = test_seq_len
    )
    
    print('mean value of best error: ', numpy.round(numpy.mean(best_test_errors), 6))
    print('min value of best error: ', numpy.round(numpy.amin(best_test_errors), 6))
    print('max value of best error: ', numpy.round(numpy.amax(best_test_errors), 6)) 

def train_sda_with_log_layer():
    pretrain_algo = 'sgd'
    pretrain_lr = 0.001    
    pretraining_epochs = 1
    pretraining_pat_epochs = 1
    
    corruption_levels = [0.1, 0.2]
    hidden_layer_sizes = [n_features/2, n_features/3]
    batch_size = 10    
    train_seq_len = 20
    test_seq_len = 40
    
    finetune_lr = 0.001
    finetune_epochs = 1
    finetune_pat_epochs = 1
    finetune_algo = 'sgd'

    # 3rd approach
    # classifier after autoencoder, long train for single model   
    trained_sda = train_sda(
        corruption_levels = corruption_levels,
        pretraining_epochs = pretraining_epochs,
        pretraining_pat_epochs = pretraining_pat_epochs,
        pretrain_lr = pretrain_lr,
        hidden_layer_sizes = hidden_layer_sizes,
        pretrain_algo = pretrain_algo,
        n_features = n_features,
        n_classes = n_classes,
        batch_size = batch_size,
        train_seq_len = train_seq_len,
        test_seq_len = test_seq_len,
        finetune_lr = finetune_lr,
        finetune_epochs = finetune_epochs,
        finetune_pat_epochs = finetune_pat_epochs,
        finetune_algo = finetune_algo
    )
    test_errors = test_sda (
        sda = trained_sda,
        test_seq_len = test_seq_len
    )
    
    print('mean value of error: ', numpy.round(numpy.mean(test_errors), 6))
    print('min value of error: ', numpy.round(numpy.amin(test_errors), 6))
    print('max value of error: ', numpy.round(numpy.amax(test_errors), 6))


if __name__ == '__main__':
    train_many_regs()
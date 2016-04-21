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
import os
import theano.tensor as T

from datetime import datetime

from logisticRegression import train_log_reg, test_log_reg
from sdA import train_sda, test_sda, finetune_sda

#define some shared constants
n_features = 75
n_classes = 39

def train_one_classifier():
    learning_rate = 0.05    
    pat_epochs = 150
    batch_size = 1000
    train_algo = 'sgd'
    global_epochs = 1
    train_seq_len = 500
    test_seq_len = 500
    
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
    approach_folder = 'long_debug_info'
    if not os.path.isdir(approach_folder):
        os.makedirs(approach_folder)
    os.chdir(approach_folder)

    debug_file_name = (('%s')%(datetime.now())).replace(':', '')
    debug_file_name = debug_file_name.replace('.','')
    debug_file = ('%s.txt')%(debug_file_name)
        
    pretrain_algo = 'sgd'
    pretrain_lr = 0.1    
    pretraining_epochs = 1
    pretraining_pat_epochs = 1000
    pretrain_attempts = 1
    
    corruption_levels = [0.1]
    hidden_layer_sizes = [200]
    activations = [T.nnet.sigmoid]
    batch_size = 1000
    train_seq_len = 500
    test_seq_len = 500
    
    finetune_lr = 0.001
    finetune_epochs = 1
    finetune_pat_epochs = 1000
    finetune_algo = 'sgd'
    finetune_attempts = 1    
    
    debug_folder = (('finetune lr %f, bs %i')%(finetune_lr, batch_size))
    if not os.path.isdir(debug_folder):
        os.makedirs(debug_folder)
    os.chdir(debug_folder)
    f = open(debug_file, 'w')
    f.write(' MODEL PARAMETERS \n')
    f.write('\n PRETRAINING \n')        
    f.write('pretrain_lr %f\n' % pretrain_lr)
    f.write('pretraining_epochs %i\n' % pretraining_epochs)
    f.write('pretraining_pat_epochs %i\n' % pretraining_pat_epochs)
    f.write('pretrain_attempts %i\n' % pretrain_attempts)
    
    f.write('\n SDA \n')        
    f.write('corruption_levels: [%s]\n' % ', '.join(map(str, corruption_levels)))
    f.write('hidden_layer_sizes:  [%s]\n' % ', '.join(map(str, hidden_layer_sizes)))
    f.write('activations:  [%s]\n' % ', '.join(map(str, activations)))
    f.write('batch_size %i\n' % batch_size)
    f.write('train_seq_len %i\n' % train_seq_len)
    f.write('test_seq_len %i\n' % test_seq_len)
    
    f.write('\n FINETUNING \n')        
    f.write('finetune_lr %f\n' % finetune_lr)
    f.write('finetune_epochs %i\n' % finetune_epochs)
    f.write('finetune_pat_epochs %i\n' % finetune_pat_epochs)
    f.write('finetune_attempts %i\n' % finetune_attempts)
    f.close()
    os.chdir('../')
    
    os.chdir('best_models')
    pretrained_sda = pickle.load(open('best_pretrain_sda.pkl'))
    os.chdir('../')

    # 3rd approach
    # classifier after autoencoder, long train for single model   
    trained_sda = finetune_sda(
        pretrained_sda = pretrained_sda,
        batch_size = batch_size,
        finetune_lr = finetune_lr,
        finetune_epochs = finetune_epochs,
        finetune_pat_epochs = finetune_pat_epochs,
        train_seq_len = train_seq_len,
        test_seq_len = test_seq_len,
        finetune_algo = finetune_algo,
        debug_folder = debug_folder,
        debug_file = debug_file,
        n_attempts = finetune_attempts
    )

    
    test_errors = test_sda (
        sda = trained_sda,
        test_seq_len = test_seq_len
    )
    
    os.chdir(debug_folder)
    f = open(debug_file, 'a')
    f.write('\n TESTING \n') 
    f.write('mean value of error: %f\n' % numpy.round(numpy.mean(test_errors), 6))
    f.write('min value of error: %f\n' % numpy.round(numpy.amin(test_errors), 6))
    f.write('max value of error: %f\n' % numpy.round(numpy.amax(test_errors), 6))
    f.close()
    os.chdir('../')
    
    os.chdir('../')


if __name__ == '__main__':
    train_sda_with_log_layer()
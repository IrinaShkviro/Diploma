# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:20:56 2016

@author: irina
"""

import os
import sys
import timeit
import gc

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import pickle

from dA import dA
from visualizer import visualize_pretraining
#from cg import pretrain_sda_cg
from sgd_with_stopping import pretrain_many_sda_sgd, finetune_log_layer_sgd
from logisticRegression import LogisticRegression
from binaryReader import BinaryReader

theano.config.exception_verbosity='high'

class SdA(object):
    def __init__(
        self,
        numpy_rng,
        n_ins,
        n_outs,
        hidden_layers_sizes,
        corruption_levels=[0.1, 0.1],
        theano_rng=None
    ):
        '''
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights
        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`
        :type n_ins: int
        :param n_ins: dimension of the input to the sdA
        
        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value
        :type n_outs: int
        :param n_outs: dimension of the output of the network
        
        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        '''

        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.n_ins=n_ins
        self.n_outs=n_outs
        
        # allocate symbolic variables for the data
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.dA_layers[-1].output           

            # Construct a denoising autoencoder that shared weights with this layer
            dA_layer = dA(
                numpy_rng=numpy_rng,
                theano_rng=theano_rng,
                input=layer_input,
                n_visible=input_size,
                n_hidden=hidden_layers_sizes[i],
                activation=T.nnet.sigmoid
            )
            
            self.dA_layers.append(dA_layer)
            self.params.extend(dA_layer.sda_params)
        
        self.logLayer = LogisticRegression(
            rng = numpy.random.RandomState(),
            input=self.dA_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )
        
        self.params.extend(self.logLayer.params)
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)
                        
def pretrain_SdA(corruption_levels,
                 pretraining_epochs,
                 pretraining_pat_epochs,
                 pretrain_lr,
                 pretrain_algo,
                 hidden_layers_sizes,
                 output_folder,
                 base_folder,
                 debug_folder,
                 debug_file,
                 n_features,
                 n_classes,
                 batch_size,
                 train_seq_len,
                 test_seq_len,
                 n_attempts = 1):
    """
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
   
    :type output_folder: string
    :param output_folder: folder for costand error graphics with results
    """    
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng = numpy.random.RandomState(),
        n_ins = n_features,
        n_outs = n_classes,
        hidden_layers_sizes = hidden_layers_sizes,
        corruption_levels = corruption_levels,
        theano_rng = None
    )
        
    #########################
    # PRETRAINING THE MODEL #
    #########################
        
    if (pretrain_algo == "sgd"):
        pretrained_sda = pretrain_many_sda_sgd(
            sda = sda,
            pretrain_lr = pretrain_lr,
            corruption_levels = corruption_levels,
            global_epochs = pretraining_epochs,
            pat_epochs = pretraining_pat_epochs,
            batch_size = batch_size,
            debug_folder = debug_folder,
            debug_file = debug_file,
            train_seq_len = train_seq_len,
            test_seq_len = test_seq_len,
            n_attempts = n_attempts
        )
    else:
        pretrained_sda = sda
    '''
        pretrained_sda = pretrain_sda_cg(
            sda=sda,
            train_names=train_names,
            window_size=window_size,
            pretraining_epochs=pretraining_epochs,
            corruption_levels=corruption_levels,
            preprocess_algo = pretrain_algo,
            read_window = read_window
        )
                             
        for i in xrange(sda.n_layers):
        visualize_pretraining(
            train_cost = pretrained_sda.dA_layers[i].train_cost_array,
            valid_error = pretrained_sda.dA_layers[i].valid_error_array,
            learning_rate = pretrain_lr,
            corruption_level = corruption_levels[i],
            n_hidden = sda.dA_layers[i].n_hidden,
            da_layer = i,
            datasets_folder = output_folder,
            base_folder = base_folder
        )
    '''
    gc.collect()
    return pretrained_sda

def finetune_sda(pretrained_sda,
                 batch_size,
                 finetune_lr,
                 finetune_epochs,
                 finetune_pat_epochs,
                 train_seq_len,
                 test_seq_len,
                 finetune_algo,
                 debug_folder,
                 debug_file,
                 n_attempts = 1):
    os.chdir('best_models')
    with open('best_sda.pkl', 'wb') as f:
        pickle.dump(pretrained_sda, f)
    os.chdir('../')
    
    os.chdir(debug_folder)
    f = open(debug_file, 'a')
    f.write('\n START FINETUNING \n')
    f.close()
    os.chdir('../')
    
    best_valid = numpy.inf
    for attempt in xrange(n_attempts):
        os.chdir('best_models')
        pretrained_sda = pickle.load(open('best_pretrain_sda.pkl'))
        os.chdir('../')
        
        # train logistic regression layer
        if finetune_algo == 'sgd':    
            finetuned_sda = finetune_log_layer_sgd(
                sda = pretrained_sda,
                batch_size = batch_size,
                finetune_lr = finetune_lr,
                global_epochs = finetune_epochs,
                pat_epochs = finetune_pat_epochs,
                train_seq_len = train_seq_len,
                test_seq_len = test_seq_len
            )
            
        else:        
            finetuned_sda = pretrained_sda
        
        os.chdir(debug_folder)
        f = open(debug_file, 'a')
        f.write('attempt: %i, ' % attempt)
        f.write('valid_error: %f\n' % finetuned_sda.logLayer.validation)
        f.close()
        os.chdir('../')
            
        if finetuned_sda.logLayer.validation < best_valid:
            best_valid = finetuned_sda.logLayer.validation
            
            os.chdir('best_models')
            with open('best_sda.pkl', 'wb') as f:
                pickle.dump(finetuned_sda, f)
            os.chdir('../')
    
    os.chdir('best_models')        
    best_sda = pickle.load(open('best_sda.pkl'))
    os.chdir('../')       
    return best_sda
        
def train_sda(corruption_levels,
              pretraining_epochs,
              pretraining_pat_epochs,
              pretrain_lr,
              hidden_layer_sizes,
              pretrain_algo,
              n_features,
              n_classes,
              batch_size,
              debug_folder,
              debug_file,
              train_seq_len,
              test_seq_len,
              finetune_lr,
              finetune_epochs,
              finetune_pat_epochs,
              finetune_algo,
              pretrain_attempts = 1,
              finetune_attempts = 1
              ):
    base_folder = 'sda_log_reg'
    pretrained_sda = pretrain_SdA(
        corruption_levels = corruption_levels,
        pretraining_epochs = pretraining_epochs,
        pretraining_pat_epochs = pretraining_pat_epochs,
        pretrain_lr = pretrain_lr,        
        pretrain_algo = pretrain_algo,
        hidden_layers_sizes = hidden_layer_sizes,
        output_folder = ('pretrain_sda_%s')%(pretrain_algo),
        base_folder = base_folder,
        debug_folder = debug_folder,
        debug_file = debug_file,
        n_features = n_features,
        n_classes = n_classes,
        batch_size = batch_size,
        train_seq_len = train_seq_len,
        test_seq_len = test_seq_len,
        n_attempts = pretrain_attempts
    )
    '''
    # train logistic regression layer
    finetuned_sda = finetune_sda(
        pretrained_sda = pretrained_sda,
        batch_size = batch_size,
        finetune_lr = finetune_lr,
        finetune_epochs = finetune_epochs,
        finetune_pat_epochs = finetune_pat_epochs,
        train_seq_len = train_seq_len,
        test_seq_len = test_seq_len,
        finetune_algo = finetune_algo,
        debug_folder = debug_folder,
        debug_file = debug_file
    )
    '''
    return pretrained_sda
    
def test_sda(sda, test_seq_len = 1):
    
    test_model = theano.function(
        inputs=[sda.x, sda.y],
        outputs=[sda.errors]
    )   
    test_reader = BinaryReader(
        isTrain=False,
        len_seqs=test_seq_len
    )
    # compute zero-one loss on test set
    test_error_array = [] 
    for test_pat in xrange(test_reader.n_files):
        test_features, test_labels = test_reader.read_several()
        test_error_array.append(test_model(
            test_features.get_value(
                borrow=True,
                return_internal_type=True
            ),
            test_labels.eval()
        ))
    gc.collect()
        
    return test_error_array
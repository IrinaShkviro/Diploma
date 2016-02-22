# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:03:53 2016

@author: irina
"""
import os
import numpy
import gc

import struct

import theano
import theano.tensor as T

class BinaryReader(object):
    def __init__(self, seqs_for_analyse, isTrain,
                 n_features = 75, data_format = 'f'):
        self.seqs = seqs_for_analyse
        
        self.n_features = n_features
        self.format = data_format
        self.format_size = struct.calcsize(self.format)
                
        self.sequence_index = 0
        # path to folder with data
        if isTrain:
            dataset = './Data/phones39/train' 
        else:
            dataset = './Data/phones39/test' 
        self.init_sequence(dataset)
    
    # read all docs in sequence
    def read_all(self):
        feature_array, labels = self.get_sequence()

        data_x = feature_array
        data_y = labels
        
        for t in range(len(self.seqs) - 1):
            #read current file
            feature_array, labels = self.get_sequence()
            
            # concatenate data in current file with data in prev files in one array
            data_x = numpy.concatenate((data_x, feature_array))
            data_y = numpy.concatenate((data_y, labels))                           
            gc.collect()
        
        set_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=True)
        set_y = T.cast(theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=True), 'int32')
        
        return (set_x, set_y) 
    
    # read one doc in sequence
    def read_next_doc(self, algo, rank=1, window=1, divide = False):    
       
        feature_array, labels = self.get_sequence()

        set_x = theano.shared(numpy.asarray(feature_array),
                                     borrow=True)
        set_y = T.cast(theano.shared(numpy.asarray(labels,
                                                   dtype=theano.config.floatX),
                                         borrow=True), 'int32')       
        return (set_x, set_y)
                   
    def init_sequence(self, dataset):
        self.sequence_files = []
        
        for f in self.seqs:
            # sequence_file - full path to each document
            sequence_file = dataset+"/"+str(f)+".binary"
            self.sequence_files.append(sequence_file)
            
    # define current file for reading
    def get_sequence(self):
        
        if self.sequence_index>=len(self.sequence_files):
            self.sequence_index = 0
            
        sequence_file = self.sequence_files[self.sequence_index]
        self.sequence_index = self.sequence_index+1
        return self.read_sequence(sequence_file)
        
    #read sequence_file and return array of features and vector with labels
    def read_sequence(self, sequence_file):
        
        filesize = os.path.getsize(sequence_file)
        n_samples = filesize/(4*(self.n_features + 1))
        
        feature_array = []
        with open(sequence_file, 'rb') as f:
            labels = []
            for sample_index in xrange(n_samples):
                #read features for sample
                sample_features = []
                for feature_number in xrange(self.n_features):                    
                    value = f.read(self.format_size)
                    sample_features.append(struct.unpack(self.format, value)[0])
                feature_array.append(sample_features)
                #read label for sample
                label = f.read(self.format_size)
                labels.append([struct.unpack(self.format, label)[0]])
            #print(numpy.concatenate((labels, feature_array), axis=1))
            f.close()
        return feature_array, labels
        
if __name__ == '__main__':
    names = ['SX56.1703']
    testReader = BinaryReader(names, True)
    testReader.read_sequence(testReader.sequence_files[0])
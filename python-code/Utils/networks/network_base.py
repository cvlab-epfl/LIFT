# network_base.py ---
#
# Filename: network_base.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Thu Feb 18 23:36:53 2016 (+0100)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C), EPFL Computer Vision Lab.

# Code:


from __future__ import print_function

import os
import pdb
import sys
# Disable future warnings (caused by theano)
import warnings
from collections import OrderedDict
from copy import deepcopy

import lasagne
import numpy as np
import theano
import theano.tensor as T

from six.moves import xrange
from Utils.lasagne_tools import loadNetwork, loadNetworkWithConfig, saveNetwork

# from optimizer import *
# from cost_n_grad import *
# from regularizor import *

warnings.simplefilter(action="ignore", category=FutureWarning)

floatX = theano.config.floatX
bUseWNN = True


class NetworkConfigBase(object):

    def __init__(self):

        self.images = ''

        self.save_dir = ''

        self.batch_size = 0  # placeholder
        self.patch_height = 0  # placeholder
        self.patch_width = 0  # placeholder
        self.num_channel = 0  # placeholder
        self.out_dim = 0  # placeholder

        # self.learning_rate = 0.01
        # self.momentum = 0.9
        self.n_epochs = 30
        # save results regardless of the validation result for this amount of
        # epochs
        self.min_epoch = 5

        self.num_siamese = 0  # will make things crash for safety
        self.modelType = 'CNN'

        self.bContinue = True  # TODO:

        self.GHH_numSUM = None
        self.GHH_numMAX = None

        self.dispFrequency = 100


class NetworkBase(object):
    '''
    classdocs
    '''

    def __init__(self, config, rng=None, **kwargs):
        '''
        Constructor

        '''
        # ------------------------------------------------------------
        # Debug setup
        self.debug_flag = bool(int(os.getenv("MLTEST_DEBUG", default="0")))

        self.name = 'NetworkBase'

        if rng is None:
            self.rng = np.random.RandomState(23455)
        else:
            self.rng = rng

        # ---------------------------------------------------------------------
        # read config
        self.config = config

        # ---------------------------------------------------------------------
        # Pre-allocate empty lists for multiple instances of the same layers
        self.layers = [None for _ in xrange(self.config.num_siamese)]

        theano.config.exception_verbosity = 'high'

    def compile4Test(self, verbose=True):

        # compile functions
        if verbose:
            print("compiling test_model_deterministic() ... ", end='\033[K')
            sys.stdout.flush()
        self.test_model_deterministic = theano.function(
            inputs=[],
            outputs=[
                lasagne.layers.get_output(self.layers[0]['output'],
                                          deterministic=True)
            ],
            givens=self.givens_test,
            on_unused_input='ignore')
        if verbose:
            print("done.")

        if verbose:
            print("compiling test_model_stochastic() ... ", end='\033[K')
            sys.stdout.flush()
        self.test_model_stochastic = theano.function(
            inputs=[],
            outputs=[
                lasagne.layers.get_output(self.layers[0]['output'],
                                          deterministic=False)
            ],
            givens=self.givens_test,
            on_unused_input='ignore')
        if verbose:
            print("done.")

    def runTest(self,
                test_data_in,
                deterministic=True,
                model_epoch="",
                verbose=True):
        '''
        The test loop

        '''

        raise NotImplementedError('OVERRIDE!')

    # ------------------------------------------------------------------------
    # -----------------Mandatory Function to be overloaded by classes
    # ------------------------------------------------------------------------

    def setupCostAndUpdates(self):
        """WRITE ME

        This is a function for setting up the update rule for learning the
        network.

        """

        self.updates = None
        self.grads = None
        self.cost_stochastic = None
        self.cost_p_stochastic = None
        self.cost_deterministic = None
        self.cost_p_deterministic = None

        raise RuntimeError(
            'Setup updates is not set! -- override this function!')

    def setup4Train(self):
        """WRITE ME

        This is a function for setting up givens which need to be fed to the
        theano train functions

        """

        self.givens_train = {}
        for idxSiam in xrange(self.config.num_siamese):
            self.givens_train[self.x[idxSiam]] = None
            self.givens_train[self.y[idxSiam]] = None

        raise RuntimeError(
            'Setup train is not set! -- override this function!')

    def setup4Test(self):
        """WRITE ME

        This is a function for setting up givens which need to be fed to the
        theano test functions

        """

        self.givens_train = {}
        for idxSiam in xrange(self.config.num_siamese):
            self.givens_test[self.x[idxSiam]] = None

        raise RuntimeError('Setup test is not set! -- override this function!')

    def learningSchedule(self):
        """WRITE ME

        This is a function for altering the learning rate and momentum of the
        network

        """

        # ---------------------------------------------------------------------
        # Change Learning rate
        cur_learning_rate = self.config.lr * \
            (0.5**(self.epoch / self.config.lr_half_interval))
        # anneal so that learning rate is halved every two epochs
        self.learning_rate.set_value(np.asarray(cur_learning_rate,
                                                dtype=floatX))

        raise RuntimeError(
            'Setup updates is not set! -- override this function!')

    def prepTrainData(self):
        """WRITE ME

        This is a function for preparing data in the format the network should
        take. Train, validation and test data should all be prepared
        properly. Data should also be shuffled.

        """

        # # The datas...
        # self.train_data_in
        # self.valid_data_in
        # self.test_data_in

        if self.config.num_siamese != 1:
            raise RuntimeError(
                'The base function only supports num_siamese == 1')

        # Crop data depending on number of train batches
        self.n_train_batches = np.int(np.floor(self.train_data_in.x.shape[0] /
                                               (self.config.batch_size)))
        self.train_data_in.x = self.train_data_in.x[
            :self.n_train_batches * self.config.batch_size]
        self.train_data_in.y = self.train_data_in.y[
            :self.n_train_batches * self.config.batch_size]

        # Reshape Into Correct Form
        self.train_data_in.x.shape = [-1, self.config.num_channel,
                                      self.config.patch_height,
                                      self.config.patch_width]

        # raise RuntimeError('Prepration of train data is not set!'
        #                    ' -- override this function!')

    def prepValidationData(self):
        """WRITE ME

        This is a function for preparing data in the format the network should
        take. Only validation data should be prepared

        """
        if self.config.num_siamese != 1:
            raise RuntimeError(
                'The base function only supports num_siamese == 1')

        # Crop validation data
        self.n_valid_batches = np.int(np.floor(self.valid_data_in.x.shape[0] /
                                               (self.config.batch_size)))
        self.valid_data_in.x = self.valid_data_in.x[
            :self.n_valid_batches * self.config.batch_size]
        self.valid_data_in.y = self.valid_data_in.y[
            :self.n_valid_batches * self.config.batch_size]

        # Reshape Into Correct Form
        self.valid_data_in.x.shape = [-1, self.config.num_channel,
                                      self.config.patch_height,
                                      self.config.patch_width]

        # raise RuntimeError('Prepration of validation data is not set!'
        #                    ' -- override this function!')

    def prepTestData(self):
        """WRITE ME

        This is a function for preparing data in the format the network should
        take. Only test data should be prepared

        """

        if self.config.num_siamese != 1:
            raise RuntimeError(
                'The base function only supports num_siamese == 1')

        # Crop test data
        self.n_test_batches = np.int(np.floor(self.test_data_in.x.shape[0] / (
            self.config.batch_size)))
        self.test_data_in.x = self.test_data_in.x[
            :self.n_test_batches * self.config.batch_size]
        self.test_data_in.y = self.test_data_in.y[
            :self.n_test_batches * self.config.batch_size]

        # Reshape Into Correct Form
        self.test_data_in.x.shape = [-1, self.config.num_channel,
                                     self.config.patch_height,
                                     self.config.patch_width]

        # raise RuntimeError('Prepration of test data is not set!'
        #                    ' -- override this function!')

    def createShuffleIdx(self):

        if self.config.num_siamese != 1:
            raise RuntimeError(
                'The base function only supports num_siamese == 1')

        n_train_batches = self.n_train_batches
        # num_siamese = self.config.num_siamese
        batch_size = self.config.batch_size

        idxShuffle = np.random.permutation(
            n_train_batches * batch_size).astype('int64')  # random shuffle all
        # # make repetitions for siamese
        # idxShuffle = np.tile(idxShuffle,(num_siamese,1)).T.flatten()
        # idxShuffle = idxShuffle*num_siamese +
        # np.tile(np.arange(num_siamese),(len(idxShuffle)/num_siamese,)) #
        # final shuffle idx

        return idxShuffle

    def copyData2Shared(self, data, idxCurDatas):
        """WRITE ME

        This is a function that copies the train data to the shared memory

        """

        raise RuntimeError('Train data transfer is not set!'
                           ' -- override this function!')

    def validationProc(self, outstring):
        """WRITE ME

        This is a function that performs validation with the validation set and
        returns the validation cost

        """

        raise RuntimeError('Validation procedure is not set!'
                           ' -- override this function!')

    def testProc(self, outstring):
        """WRITE ME

        This is a function that performs test with the test set and returns the
        test cost

        """

        raise RuntimeError('Test procedure is not set!'
                           ' -- override this function!')

    def prepCrossValidatationData(self):
        """WRITE ME

        This is a function for preparing data in the format the network should
        take for cross validation. Only validation and test data should be
        prepared

        """

        raise RuntimeError('Preparation of cross validation data is not set!'
                           ' -- override this function!')

    # ------------------------------------------------------------------------
    # -----------------Optional Function to be overloaded by classes using thi
    # ------------------------------------------------------------------------

    def setupSpecific(self):
        """WRITE ME

        This is a function for any specific setups that are necessary for
        additional theano functions specific to the network

        """

        print('Warning: Setup specific is not set! -- this might be okay')

    def compileSpecific(self, verbose=True):
        """WRITE ME

        This is a function for any compiles additional theano functions
        specific to the network

        """

        if verbose:
            print('Warning: Compile specific is not set!'
                  ' -- this might be okay')

    def computeTrainRes(self, cur_train_res):
        """WRITE ME

        This is a function that computes the training result to display
        (default classif error)

        """
        cur_out = cur_train_res[2]
        batch_size = self.config.batch_size

        cur_res = float(np.sum(np.argmax(cur_out,
                                         axis=1) !=
                               np.argmax(self.train_y[0].get_value(),
                                         axis=1))) / float(batch_size)

        return cur_res

    def postTrainProc(self):
        """WRITE ME

        This is a function that performs procedures after training (to display
        additional information)

        """

        print('Warning: Post validation procedure is not set!'
              ' -- this might be okay')

    def postValidationProc(self):
        """WRITE ME

        This is a function that performs procedures after validation (to
        display additional information)

        """

        print('Warning: Post validation procedure is not set!'
              ' -- this might be okay')

    def customInit(self):
        """WRITE ME

        This is a function for any custom initializations to be carried out
        before learning

        """

        print('Warning: Custom initialization is not set!'
              ' -- this might be okay')

#
# network_base.py ends here

# eccv_base.py ---
#
# Filename: eccv_base.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Thu Feb 18 14:54:57 2016 (+0100)
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

import hashlib
import os
import sys
import warnings
from datetime import timedelta

import lasagne
import numpy as np
import six
import theano
import theano.tensor as T
from flufl.lock import Lock
from lasagne.layers import get_all_params, get_output, get_output_shape
from parse import parse

from Utils.dump_tools import loadh5, saveh5
from Utils.lasagne_tools import (createXYZMapLayer, loadNetwork,
                                 loadNetworkWithConfig)
from Utils.networks.network_base import NetworkBase, NetworkConfigBase

# Disable future warnings (caused by theano)
warnings.simplefilter(action="ignore", category=FutureWarning)

floatX = theano.config.floatX
bUseWNN = True


class ECCVNetworkConfigBase(NetworkConfigBase):

    def __init__(self):

        # Call base init function
        super(ECCVNetworkConfigBase, self).__init__()

        # Additional Inits
        self.num_siamese = 2  # single siamese
        self.numericGradType = None  # no numeric grads
        self.gradType = 'SimpleCost'  # Gradient from simply
        # derivating the cost

        # Display Frequency
        self.dispFrequency = 10


class ECCVNetworkBase(NetworkBase):
    '''
    classdocs
    '''

    def __init__(self, config, rng=None, verbose=True, **kwargs):

        # Call base init function
        super(ECCVNetworkBase, self).__init__(config, rng, **kwargs)

        # Additional Inits
        # Look Up Tables stored as dictionaries
        self.LUT = {}

        # --------------------------------------------------------------------
        # Theano Variables

        # N x D x P x P : data (P is the patch size, D is the number
        # of channels)
        self.x = [None for _ in six.moves.xrange(self.config.num_siamese)]
        # N x O : target value (O is the number of output)
        self.y = [None for _ in six.moves.xrange(self.config.num_siamese)]
        # N x O : target value (O is the number of output)
        self.ID = [None for _ in six.moves.xrange(self.config.num_siamese)]
        self.pos = [None for _ in six.moves.xrange(self.config.num_siamese)]
        self.angle = [None for _ in six.moves.xrange(self.config.num_siamese)]

        for idxSiam in six.moves.xrange(self.config.num_siamese):
            # N x D x P x P : data (P is the patch size, D is the number of
            # channels)
            self.x[idxSiam] = T.tensor4('x_' + str(idxSiam), dtype=floatX)
            # self.x[idxSiam] = T.TensorType(floatX, ((False,)*5))('x_' +
            #                                                      str(idxSiam))
            # # N x D x P x P x S : data (P is the patch size, D is the number
            # # of channels, S is the number of Scales)
            # N x O : target label in one-hot coding
            self.y[idxSiam] = T.vector('y_' + str(idxSiam), dtype=floatX)
            # N x O : target label in one-hot coding
            self.ID[idxSiam] = T.vector('ID_' + str(idxSiam), dtype='int64')
            self.pos[idxSiam] = T.matrix('pos_' + str(idxSiam), dtype=floatX)
            self.angle[idxSiam] = T.vector(
                'angle_' + str(idxSiam), dtype=floatX)

        self.buildLayers(verbose=verbose, **kwargs)

    def buildLayers(self, verbose=True):
        """
        This is a function for creating layer args to be used in instantiation

        For mnist, we use the basic LeNet5 example in Theano tutorial

        """

        raise RuntimeError('buildLayers not overwritten!')

    def runTest(self,
                test_data_in,
                test_mode=None,
                deterministic=True,
                model_epoch="",
                verbose=True,
                network_weights=None):
        """The test fuction.

        This function returns the orientation and the descriptor extracted for
        a given test_data.

        Notes
        -----
        Bare in mind that test_data_in should only be of one siamese!

        """

        # Read parameters
        batch_size = self.config.batch_size
        save_dir = self.config.save_dir

        # Load the model
        if network_weights is None:
            if os.path.exists(save_dir + model_epoch):

                # for each layer of the model
                for idxSiam in six.moves.xrange(self.config.num_siamese):
                    save_file_name = save_dir + model_epoch + "model"
                    loadNetwork(self.layers[idxSiam], save_file_name)

            else:
                print(("PATH:", save_dir + model_epoch))
                raise NotImplementedError(
                    "I DON'T HAVE THE LEARNED MODEL READY!")
        else:
            # for each layer of the model
            for idxSiam in six.moves.xrange(self.config.num_siamese):
                # save_file_name = save_dir + model_epoch + "model"
                save_file_name = network_weights
                loadNetwork(self.layers[idxSiam], save_file_name)

        # Test data preparation. Note that we are NOT using the prep function
        # which re-orders the data in siamese friendly way for learning.
        n_batches = np.int(np.ceil(
            len(test_data_in.ID) /
            float(batch_size)
        ))

        descs = [None] * n_batches
        oris = [None] * n_batches
        # For each batch, compute the distances.
        for idx_batch in six.moves.xrange(n_batches):
            if verbose:
                print('\rTesting {} / {}'.format(idx_batch + 1, n_batches),
                      end='\033[K')
                sys.stdout.flush()

            # Alter batch size for last epoch
            if idx_batch == n_batches - 1:
                cur_batch_size = (len(test_data_in.ID) -
                                  idx_batch * batch_size)
            else:
                cur_batch_size = batch_size

            # Current data indices
            idxCurDatas = np.arange(cur_batch_size, dtype='int64')
            idxCurDatas += batch_size * idx_batch

            # Copy Validation Data to Shared, but with num_siamese=1. In case
            # of the testing data, we do not shuffle or prepare it in siamese
            # format.
            self.copyData2Shared(src=test_data_in, dst='test',
                                 idx=idxCurDatas, num_siamese=1)

            cur_descs = None
            cur_oris = None
            if deterministic:
                if test_mode == "desc":
                    # print("running DESC")
                    cur_descs = self.test_model_deterministic()[-1]
                if test_mode == "ori":
                    # print("running ORI")
                    cur_oris = self.test_ori_deterministic()[-1]
            else:
                if test_mode == "desc":
                    # print("running DESC")
                    cur_descs = self.test_model_stochastic()[-1]
                if test_mode == "ori":
                    # print("running ORI")
                    cur_oris = self.test_ori_stochastic()[-1]

            if test_mode == "desc":
                descs[idx_batch] = np.asarray(cur_descs)[:cur_batch_size]
            if test_mode == "ori":
                oris[idx_batch] = np.asarray(cur_oris)[:cur_batch_size]

        if test_mode == "desc":
            descs = np.concatenate(descs)
        if test_mode == "ori":
            oris = np.concatenate(oris)

        print(' ... done.')

        return descs, oris

    def runTest4Image(self,
                      image,
                      deterministic=True,
                      model_epoch="",
                      verbose=True,
                      network_weights=None):
        '''WRITE ME

        The test loop

        '''

        save_dir = self.config.save_dir

        # ---------------------------------------------------------------------
        # Testing Loop
        if verbose:
            print('testing...')

        # ---------------------------------------------------------------------
        # Load the model
        if network_weights is None:
            if os.path.exists(save_dir + model_epoch):

                # for each layer of the model
                for idxSiam in six.moves.xrange(self.config.num_siamese):
                    save_file_name = save_dir + model_epoch + "model"
                    loadNetwork(self.layers[idxSiam], save_file_name)

            else:
                print(("PATH:", save_dir + model_epoch))
                raise NotImplementedError(
                    "I DON'T HAVE THE LEARNED MODEL READY!")
        else:
            # for each layer of the model
            for idxSiam in six.moves.xrange(self.config.num_siamese):
                # save_file_name = save_dir + model_epoch + "model"
                save_file_name = network_weights
                loadNetwork(self.layers[idxSiam], save_file_name)

        # Copy test data to share memory (test memory), we don't care about the
        # labels here
        val = image.reshape([1, 1, image.shape[0], image.shape[1]])
        if self.config.bNormalizeInput:
            val = (val - self.config.mean_x) / self.config.std_x
        else:
            val /= np.cast[floatX](255.0)
        self.test_x[0].set_value(val)
        # for idxSiam in six.moves.xrange(num_siamese):
        #     self.test_x[idxSiam].set_value(image)

        if deterministic:
            # print("running KP")
            return self.test_scoremap_deterministic()[-1]
        else:
            # print("running KP")
            return self.test_scoremap_stochastic()[-1]

    def setup4Test(self):
        """WRITE ME

        This is a function for setting up givens which need to be fed to the
        theano test functions

        """

        # ---------------------------------------------------------------------
        # Allocate Theano Shared Variables (allocated on the GPU)
        batch_size = self.config.batch_size
        patch_width = self.config.patch_width
        patch_height = self.config.patch_height
        num_channel = self.config.num_channel
        # out_dim = self.config.out_dim

        self.test_x = [None for _ in six.moves.xrange(self.config.num_siamese)]
        self.test_y = [None for _ in six.moves.xrange(self.config.num_siamese)]
        self.test_ID = [None for _ in six.moves.xrange(
            self.config.num_siamese)]
        self.test_pos = [None for _ in six.moves.xrange(
            self.config.num_siamese)]
        self.test_angle = [
            None for _ in six.moves.xrange(self.config.num_siamese)]

        for idxSiam in six.moves.xrange(self.config.num_siamese):
            self.test_x[idxSiam] = theano.shared(
                np.zeros(
                    (batch_size, num_channel, patch_height, patch_width),
                    dtype=floatX),
                name='test_x_' + str(idxSiam))
            self.test_y[idxSiam] = theano.shared(
                np.zeros(
                    (batch_size, ),
                    dtype=floatX),
                name='test_y_' + str(idxSiam))
            self.test_ID[idxSiam] = theano.shared(
                np.zeros(
                    (batch_size, ),
                    dtype='int64'),
                name='test_ID_' + str(idxSiam))
            self.test_pos[idxSiam] = theano.shared(
                np.zeros(
                    (batch_size, 3),
                    dtype=floatX),
                name='test_pos_' + str(idxSiam))
            self.test_angle[idxSiam] = theano.shared(
                np.zeros(
                    (batch_size, ),
                    dtype=floatX),
                name='test_angle_' + str(idxSiam))

        # ---------------------------------------------------------------------
        # Compile Functions for Testing

        # setup givens (TODO: consider using the macro batch stuff)
        self.givens_test = {}
        for idxSiam in six.moves.xrange(self.config.num_siamese):
            self.givens_test[self.x[idxSiam]] = self.test_x[idxSiam]
            self.givens_test[self.y[idxSiam]] = self.test_y[idxSiam]
            self.givens_test[self.ID[idxSiam]] = self.test_ID[idxSiam]
            self.givens_test[self.pos[idxSiam]] = self.test_pos[idxSiam]
            self.givens_test[self.angle[idxSiam]] = self.test_angle[idxSiam]

    # def prepValidationData(self):
    #     """WRITE ME


    #     This is a function for preparing data in the format the network should
    #     take. Train, validation and test data should all be prepared
    #     properly. Data should also be shuffled.

    #     """

    #     # Find the number of pairs we can get without repetition
    #     max_id = self.lookup('self.valid_data_in.ID.max()')
    #     num_possible_pair = 0
    #     for cur_id in six.moves.xrange(max_id + 1):
    #         if cur_id % 1000 == 0:
    #             print('\rPreparing valid data '
    #                   '{} / {}'.format(cur_id + 1, max_id + 1),
    #                   end='\033[K')
    #             sys.stdout.flush()
    #         # compute the number of possible pairs
    #         num_possible_pair += len(self.ID2IdxArray('self.valid_data_in.ID',
    #                                                   cur_id)) // 2
    #     print('\rPreparing valid data {} / {} ... done'.format(
    #         max_id + 1, max_id + 1))

    #     # retrieve neg per pos
    #     neg_per_pos = getattr(self.config, 'nNegPerPos', 1)

    #     # Divide with batch size to determine number of training batches
    #     self.n_valid_pairs = num_possible_pair * (1 + neg_per_pos)
    #     self.n_valid_batches = np.int(np.ceil(
    #         self.n_valid_pairs /
    #         float(self.config.batch_size)
    #     ))

    #     # Check if we have a shuffle idx ready
    #     if len(self.config.trainSetList) == 1:
    #         dump_dir = self.config.patch_dump
    #     else:
    #         dump_dir = self.config.save_dir
    #     shuffle_file = (dump_dir + 'valid_shuffle' +
    #                     str(self.n_valid_pairs) +
    #                     str(neg_per_pos) + '.h5')
    #     check_lock_file = ('.locks/' +
    #                        hashlib.md5(shuffle_file.encode()).hexdigest() +
    #                        '.lock')
    #     check_lock = Lock(check_lock_file)
    #     check_lock.lifetime = timedelta(days=2)
    #     check_lock.lock()
    #     if not os.path.exists(shuffle_file):
    #         print('Creating shuffles for validation')
    #         # Create a Shuffle Idx to reorder the validation data
    #         shuffleIdx = self.createShuffleWithID('self.valid_data_in.ID',
    #                                               self.n_valid_pairs,
    #                                               neg_per_pos=neg_per_pos)
    #         saveh5({'shuffleIdx': shuffleIdx}, shuffle_file)
    #     else:
    #         print('Loading pre-existing shuffles for validation')
    #         shuffleIdx = loadh5(shuffle_file)['shuffleIdx']
    #     check_lock.unlock()

    #     # Restrict to a reasonable size
    #     max_val_n_batch = getattr(self.config, 'dMaxValidBatch', 1000)
    #     batch_size = self.config.batch_size
    #     num_siamese = self.config.num_siamese
    #     idx2keep = min(len(shuffleIdx), num_siamese * batch_size *
    #                    max_val_n_batch)
    #     shuffleIdx = shuffleIdx[:idx2keep]

    #     # Store the shuffle idx
    #     self.valid_shuffle_idx = shuffleIdx

    #     # # Reorder the validation data
    #     # self.valid_data_in.x = self.valid_data_in.x[shuffleIdx]
    #     # self.valid_data_in.y = self.valid_data_in.y[shuffleIdx]
    #     # self.valid_data_in.ID = self.valid_data_in.ID[shuffleIdx]
    #     # self.valid_data_in.pos = self.valid_data_in.pos[shuffleIdx]
    #     # self.valid_data_in.angle = self.valid_data_in.angle[shuffleIdx]

    # def prepTestData(self):
    #     """WRITE ME

    #     This is a function for preparing data in the format the network should
    #     take. Train, validation and test data should all be prepared
    #     properly. Data should also be shuffled.

    #     """

    #     # Find the number of pairs we can get without repetition
    #     max_id = self.lookup('self.test_data_in.ID.max()')
    #     num_possible_pair = 0
    #     for cur_id in six.moves.xrange(max_id + 1):
    #         if cur_id % 1000 == 0:
    #             print('\rPreparing test data '
    #                   '{} / {}'.format(cur_id + 1, max_id + 1),
    #                   end='\033[K')
    #             sys.stdout.flush()
    #         # compute the number of possible pairs
    #         num_possible_pair += len(self.ID2IdxArray('self.test_data_in.ID',
    #                                                   cur_id)) // 2
    #     print('\rPreparing test data {} / {} ... done'.format(
    #         max_id + 1, max_id + 1))

    #     # retrieve neg per pos
    #     neg_per_pos = getattr(self.config, 'nNegPerPos', 1)

    #     # Divide with batch size to determine number of training batches
    #     self.n_test_pairs = num_possible_pair * (1 + neg_per_pos)
    #     self.n_test_batches = np.int(np.ceil(
    #         self.n_test_pairs /
    #         float(self.config.batch_size)
    #     ))

    #     # Check if we have a shuffle idx ready
    #     if len(self.config.trainSetList) == 1:
    #         dump_dir = self.config.patch_dump
    #     else:
    #         dump_dir = self.config.save_dir
    #     shuffle_file = (dump_dir + 'test_shuffle' +
    #                     str(self.n_test_pairs) +
    #                     str(neg_per_pos) + '.h5')
    #     check_lock_file = ('.locks/' +
    #                        hashlib.md5(shuffle_file.encode()).hexdigest() +
    #                        '.lock')
    #     check_lock = Lock(check_lock_file)
    #     check_lock.lifetime = timedelta(days=2)
    #     check_lock.lock()
    #     if not os.path.exists(shuffle_file):
    #         print('Creating shuffles for test')
    #         # Create a Shuffle Idx to reorder the test data
    #         shuffleIdx = self.createShuffleWithID('self.test_data_in.ID',
    #                                               self.n_test_pairs,
    #                                               neg_per_pos=neg_per_pos)
    #         saveh5({'shuffleIdx': shuffleIdx}, shuffle_file)
    #     else:
    #         print('Loading pre-existing shuffles for test')
    #         shuffleIdx = loadh5(shuffle_file)['shuffleIdx']
    #     check_lock.unlock()

    #     # Restrict to a reasonable si
    #     max_test_n_batch = getattr(self.config, 'dMaxTestBatch', 1000)
    #     batch_size = self.config.batch_size
    #     num_siamese = self.config.num_siamese
    #     idx2keep = min(len(shuffleIdx), num_siamese * batch_size *
    #                    max_test_n_batch)
    #     shuffleIdx = shuffleIdx[:idx2keep]

    #     # Store the shuffle idx
    #     self.test_shuffle_idx = shuffleIdx

    #     # # Reorder the test data
    #     # self.test_data_in.x = self.test_data_in.x[shuffleIdx]
    #     # self.test_data_in.y = self.test_data_in.y[shuffleIdx]
    #     # self.test_data_in.ID = self.test_data_in.ID[shuffleIdx]
    #     # self.test_data_in.pos = self.test_data_in.pos[shuffleIdx]
    #     # self.test_data_in.angle = self.test_data_in.angle[shuffleIdx]

    def copyData2Shared(self, src, dst, idx, num_siamese=None):
        """Copies data to shared memory.

        Parameters
        ----------
        src: data object
            Input data object to be coppied

        dst: str
            Prefix of the shared variable. E.g. 'train', 'test'

        idx: ndarray of int
            Indices of the data to be copied.

        num_siamese: int, optional
            The number of siamese that was used to format the data. Note that
            in case of using this for the testing function, we do NOT form it
            as siamese, so this should be set to 1. Otherwise, just don't use
            this parameter so that it is set to the number of siamese of the
            network as default.

        """

        if num_siamese is None:
            num_siamese = self.config.num_siamese
        batch_size = self.config.batch_size

        # Copy Train data to Shared (Considering pairs are stuck together)
        for idxSiam in six.moves.xrange(num_siamese):
            idxCurSiam = idx[(np.arange(len(idx) / num_siamese) *
                              num_siamese + idxSiam).astype(int)]

            # If train, make sure we fill all the data
            if dst == 'train':
                assert len(idx) / num_siamese == batch_size

            # For x
            val = np.zeros((batch_size,) + src.x.shape[1:], dtype=floatX)
            val[:len(idxCurSiam)] = np.asarray(
                [np.asarray(src.x[idxC]) for idxC in idxCurSiam],
                dtype=floatX)
            if self.config.bNormalizeInput:
                val = (val - self.config.mean_x) / self.config.std_x
            else:
                val /= np.cast[floatX](255.0)
            getattr(self, dst + '_x')[idxSiam].set_value(val)
            # For y
            val = np.zeros((batch_size,) + src.y.shape[1:], dtype=floatX)
            val[:len(idxCurSiam)] = np.asarray(
                [np.asarray(src.y[idxC]) for idxC in idxCurSiam],
                dtype=floatX)
            getattr(self, dst + '_y')[idxSiam].set_value(val)
            # For ID
            val = np.zeros((batch_size,) + src.ID.shape[1:], dtype='int64')
            val[:len(idxCurSiam)] = np.asarray(
                [np.asarray(src.ID[idxC]) for idxC in idxCurSiam],
                dtype='int64')
            getattr(self, dst + '_ID')[idxSiam].set_value(val)
            # For pos
            val = np.zeros((batch_size,) + src.pos.shape[1:], dtype=floatX)
            val[:len(idxCurSiam)] = np.asarray(
                [np.asarray(src.pos[idxC]) for idxC in idxCurSiam],
                dtype=floatX)
            getattr(self, dst + '_pos')[idxSiam].set_value(val)
            # For angle
            val = np.zeros((batch_size,) + src.angle.shape[1:], dtype=floatX)
            val[:len(idxCurSiam)] = np.asarray(
                [np.asarray(src.angle[idxC]) for idxC in idxCurSiam],
                dtype=floatX)
            getattr(self, dst + '_angle')[idxSiam].set_value(val)

        # if self.debug_flag:
        #     import pdb
        #     pdb.set_trace()

    def compileSpecific(self, verbose=True):

        if verbose:
            print("compiling test_scoremap_deterministic() ... ", end='\033[K')
            sys.stdout.flush()
        self.test_scoremap_deterministic = theano.function(
            inputs=[],
            outputs=[
                get_output(self.layers[0]['kp-scoremap'],
                           deterministic=True)
            ],
            givens=self.givens_test,
            on_unused_input='ignore')
        if verbose:
            print("done.")

        if verbose:
            print("compiling test_scoremap_stochastic() ... ", end='\033[K')
            sys.stdout.flush()
        self.test_scoremap_stochastic = theano.function(
            inputs=[],
            outputs=[
                get_output(self.layers[0]['kp-scoremap'],
                           deterministic=False)
            ],
            givens=self.givens_test,
            on_unused_input='ignore')
        if verbose:
            print("done.")

        # ---------------------------------------------------------------------
        # Function for testing
        if verbose:
            print("compiling test_ori_stochastic() ... ",
                  end='\033[K')
            sys.stdout.flush()
        self.test_ori_stochastic = theano.function(
            inputs=[],
            outputs=[
                get_output(self.layers[0]['ori-output'],
                           deterministic=False)
            ],
            givens=self.givens_test,
            on_unused_input='ignore')
        if verbose:
            print("done.")

        if verbose:
            print("compiling test_ori_deterministic() ... ",
                  end='\033[K')
            sys.stdout.flush()
        self.test_ori_deterministic = theano.function(
            inputs=[],
            outputs=[
                get_output(self.layers[0]['ori-output'],
                           deterministic=True)
            ],
            givens=self.givens_test,
            on_unused_input='ignore')
        if verbose:
            print("done.")

        return

#
# eccv_base.py ends here

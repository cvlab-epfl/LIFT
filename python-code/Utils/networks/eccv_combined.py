# siamese_orientation.py ---
#
# Filename: siamese_orientation.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Thu Apr 30 14:48:52 2015 (+0200)
# Version:
# Package-Requires: ()
# Last-Updated: Tue Sep 20 15:52:32 2016 (+0200)
#           By: kwang
#     Update #: 1465
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
#
# Copyright (C), EPFL Computer Vision Lab.
#
#

# Code:

from __future__ import print_function

import importlib
import warnings
from collections import OrderedDict

import numpy as np
import six
import theano
from lasagne.layers import FlattenLayer, InputLayer

from Utils.lasagne_tools import (createConvtScaleSpaceLayer, createResizeLayer,
                                 createXYZCropLayer, createXYZTCropLayer)
from Utils.networks.eccv_base import ECCVNetworkBase, ECCVNetworkConfigBase

# Disable future warnings (caused by theano)
warnings.simplefilter(action="ignore", category=FutureWarning)

floatX = theano.config.floatX
bUseWNN = True


class NetworkConfig(ECCVNetworkConfigBase):

    def __init__(self):

        # Call base init function
        super(NetworkConfig, self).__init__()


class Network(ECCVNetworkBase):
    '''
    classdocs
    '''

    def __init__(self, config, rng=None, bTestWholeImage=False, verbose=True):

        # Call base init function
        super(Network, self).__init__(config,
                                      rng,
                                      bTestWholeImage=bTestWholeImage,
                                      verbose=verbose)

    def buildLayers(self, bTestWholeImage=False, verbose=True):
        """
        This is a function for creating layer args to be used in instantiation

        For mnist, we use the basic LeNet5 example in Theano tutorial

        """
        if verbose:
            print(' --------------------------------------------------- ')
            print(' Build Layers ')
            print(' --------------------------------------------------- ')

        for idxSiam in six.moves.xrange(self.config.num_siamese):
            self.layers[idxSiam] = OrderedDict()

            # 2D raw input
            self.layers[idxSiam]['input_raw_2d'] \
                = InputLayer(
                    (self.config.batch_size, 1, self.config.patch_height,
                     self.config.patch_width),
                    input_var=self.x[idxSiam],
                    name='input_raw_2d')

            # Build Layers for Kp
            self.buildLayersKp(idxSiam, resize=(
                not bTestWholeImage), verbose=verbose)

            # Build Layers for orientation
            self.buildLayersOri(idxSiam, verbose)

            # Build Layers for descriptor
            self.buildLayersDesc(idxSiam, verbose)

            # Pass through the desc-output laer results to output
            self.layers[idxSiam]['output'] = FlattenLayer(
                self.layers[idxSiam]['desc-output'], 2
            )

    def buildLayersKp(self, idxSiam, resize, verbose=True):

        # ---------------------------------------------------------------------
        # Prepare input (resize & scale space)

        # resize
        if resize:
            kp_input_name = 'kp-input_resized_2d'
            # resize to kp patch size
            resize_ratio = float(self.config.nPatchSizeKp) / \
                float(self.config.nPatchSize)
            self.layers[idxSiam]['kp-input_resized_2d'] = createResizeLayer(
                self.layers[idxSiam]['input_raw_2d'],
                resize_ratio,
                name='kp-input_resized_2d')
        else:
            kp_input_name = 'input_raw_2d'

        # scale space
        assert np.max(self.config.fScaleList) == self.config.fScaleList[-1]
        assert np.max(self.config.fScaleList) == self.config.fMaxScale
        resize_ratio_list = [v / self.config.fScaleList[-1]
                             for v in self.config.fScaleList]
        self.layers[idxSiam]['kp-input'] = createConvtScaleSpaceLayer(
            self.layers[idxSiam][kp_input_name],
            resize_ratio_list,
            name='kp-input')
        if verbose and idxSiam == 0:
            print(' -- resize info: ' +
                  np.array2string(np.asarray(resize_ratio_list), precision=2))

        # ---------------------------------------------------------------------
        # Build the actual layer configuration

        # Import module
        sDetector = getattr(self.config, 'sDetector', 'tilde')
        detector_module = importlib.import_module(
            'Utils.networks.detector.' + sDetector)

        # Run build
        detector_module.build(self, idxSiam, verbose)

    def buildLayersOri(self, idxSiam, verbose=True):

        # ---------------------------------------------------------------------
        # Prepare input (resize & scale space)

        # # crop according to the keypoint component's output
        # scoremap_shape = get_output_shape(self.layers[0]['kp-scoremap_cut'])
        # num_scale = len(self.config.fScaleList)
        # # scale_space_min changes according to scoremap_shape
        # num_scale_after = scoremap_shape[4]
        # new_min_idx = (num_scale - num_scale_after) / 2
        # scale_space_min = self.config.fScaleList[new_min_idx]

        self.layers[idxSiam]['ori-input'] = createXYZCropLayer(
            self.layers[idxSiam]['input_raw_2d'],
            self.layers[idxSiam]['kp-output'],
            max(self.config.fScaleList),
            self.config.nDescInputSize,
            name='ori-input',
        )

        # ---------------------------------------------------------------------
        # Build the actual layer configuration

        # Import module
        sOrientation = getattr(self.config, 'sOrientation', 'cvpr16')
        orientation_module = importlib.import_module(
            'Utils.networks.orientation.' + sOrientation)

        # Run build
        orientation_module.build(self, idxSiam, verbose)

    def buildLayersDesc(self, idxSiam, verbose=True):

        # ---------------------------------------------------------------------
        # Prepare input (resize & scale space)

        # # crop according to the keypoint component's output
        # scoremap_shape = get_output_shape(self.layers[0]['kp-scoremap_cut'])
        # num_scale = len(self.config.fScaleList)
        # # scale_space_min changes according to scoremap_shape
        # num_scale_after = scoremap_shape[4]
        # new_min_idx = (num_scale - num_scale_after) / 2
        # scale_space_min = self.config.fScaleList[new_min_idx]

        self.layers[idxSiam]['desc-input'] = createXYZTCropLayer(
            self.layers[idxSiam]['input_raw_2d'],
            self.layers[idxSiam]['kp-output'],
            self.layers[idxSiam]['ori-output'],
            max(self.config.fScaleList),
            self.config.nDescInputSize,
            name='desc-input',
        )

        # ---------------------------------------------------------------------
        # Build the actual layer configuration

        # Import module
        sDescriptor = getattr(self.config, 'sDescriptor', 'norm_patch')
        descriptor_module = importlib.import_module(
            'Utils.networks.descriptor.' + sDescriptor)

        # Run build
        descriptor_module.build(self, idxSiam, verbose)

#
# siamese_orientation.py ends here

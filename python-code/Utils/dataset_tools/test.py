# test.py ---
#
# Filename: test.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Thu Feb 18 17:16:20 2016 (+0100)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary: Dataset class for forming the data into a data_obj we
# can use with our learning framework.
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

import cv2
import numpy as np

from Utils.custom_types import pathConfig
from Utils.dataset_tools.helper import load_patches
from Utils.dump_tools import loadh5
from Utils.kp_tools import IDX_ANGLE, loadKpListFromTxt

number_of_process = 20


class data_obj(object):
    ''' Dataset Object class.

    Implementation of the dataset object
    '''

    def __init__(self, param, image_file_name, kp_file_name):

        # Set parameters
        self.out_dim = 1        # a single regressor output (TODO)

        # Load data
        # self.x = None           # data (patches) to be used for learning [N,
        #                         # channel, w, h]
        # self.y = None           # label/target to be learned
        # self.ID = None          # id of the data for manifold regularization
        self.load_data(param, image_file_name, kp_file_name)

        # Set parameters
        self.num_channel = self.x.shape[1]  # single channel image
        # patch width == patch height (28x28)
        self.patch_height = self.x.shape[2]
        self.patch_width = self.x.shape[3]

    def load_data(self, param, image_file_name, kp_file_name):

        print(' --------------------------------------------------- ')
        print(' Test Data Module ')
        print(' --------------------------------------------------- ')

        pathconf = pathConfig()
        pathconf.setupTrain(param, 0)

        cur_data = self.load_data_for_set(
            pathconf, param, image_file_name, kp_file_name)
        self.x = cur_data[0]
        self.y = cur_data[1]
        self.ID = cur_data[2]
        self.pos = cur_data[3]
        self.angle = cur_data[4]
        self.coords = cur_data[5]

        print(' -- Loading finished')

    def load_data_for_set(self, pathconf, param,
                          image_file_name, kp_file_name):

        bUseColorImage = getattr(param.patch, "bUseColorImage", False)
        if not bUseColorImage:
            # If there is not gray image, load the color one and convert to
            # gray
            if os.path.exists(image_file_name.replace(
                    "image_color", "image_gray"
            )):
                img = cv2.imread(image_file_name.replace(
                    "image_color", "image_gray"
                ), 0)
                assert len(img.shape) == 2
            else:
                # read the image
                img = cv2.cvtColor(cv2.imread(image_file_name),
                                   cv2.COLOR_BGR2GRAY)
            in_dim = 1

        else:
            img = cv2.imread(image_file_name)
            in_dim = 3
            assert(img.shape[-1] == in_dim)

        bTestWithTestMeanStd = getattr(
            param.validation, 'bTestWithTestMeanStd', False)
        if bTestWithTestMeanStd:
            img = img - np.mean(img)
            mean_std_dict = loadh5(pathconf.result + 'mean_std.h5')
            img = img + mean_std_dict['mean_x']

            print("Test image has range {} to {} after "
                  "transforming to the test domain"
                  "".format(np.min(img), np.max(img)))

        # load the keypoint informations
        kp = np.asarray(loadKpListFromTxt(kp_file_name))

        # Assign dummy values to y, ID, angle
        y = np.zeros((len(kp),))
        ID = np.zeros((len(kp),), dtype='int64')
        # angle = np.zeros((len(kp),))
        angle = np.pi / 180.0 * kp[:, IDX_ANGLE]  # store angle in radians

        # load patches with id (drop out of boundary)
        bPerturb = False
        fPerturbInfo = np.zeros((3,))
        dataset = load_patches(img, kp, y, ID, angle, param.patch.fRatioScale,
                               param.patch.fMaxScale, param.patch.nPatchSize,
                               param.model.nDescInputSize, in_dim, bPerturb,
                               fPerturbInfo, bReturnCoords=True)

        x = dataset[0]
        y = dataset[1]
        ID = dataset[2]
        pos = dataset[3]
        angle = dataset[4]
        coords = dataset[5]

        return x, y, ID, pos, angle, coords


#
# test.py ends here

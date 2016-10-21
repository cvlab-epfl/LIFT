# compute_detector.py ---
#
# Filename: compute_detector.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Thu Feb 18 19:48:09 2016 (+0100)
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
import sys
import time
from copy import deepcopy

import cv2
import h5py
import numpy as np

from Utils.custom_types import paramGroup, paramStruct, pathConfig
from Utils.dump_tools import loadh5
from Utils.filter_tools import apply_learned_filter_2_image_no_theano
from Utils.kp_tools import XYZS2kpList, get_XYZS_from_res_list, saveKpListToTxt
from Utils.sift_tools import recomputeOrientation
from Utils.solvers import TestImage

floatX = 'float32'

# ------------------------------------------
# Main routine
if __name__ == '__main__':

    total_time = 0

    # ------------------------------------------------------------------------
    # Read arguments
    if len(sys.argv) < 6 or len(sys.argv) > 9:
        raise RuntimeError('USAGE: python compute_detector.py '
                           '<config_file> '
                           '<image_file> <output_file> '
                           '<bSavePng> <bUseTheano> '
                           '<bPrintTime/optional> '
                           '<model_dir/optional> '
                           '<num_keypoint/optional> ')

    config_file = sys.argv[1]
    image_file_name = sys.argv[2]
    output_file = sys.argv[3]
    bSavePng = bool(int(sys.argv[4]))
    bUseTheano = bool(int(sys.argv[5]))
    if len(sys.argv) >= 7:
        bPrintTime = bool(int(sys.argv[6]))
    else:
        bPrintTime = False
    if len(sys.argv) >= 8:
        model_dir = sys.argv[7]
    else:
        model_dir = None
    if len(sys.argv) >= 9:
        num_keypoint = int(sys.argv[8])
    else:
        num_keypoint = 1000


    # ------------------------------------------------------------------------
    # Setup and load parameters
    param = paramStruct()
    param.loadParam(config_file, verbose=True)
    pathconf = pathConfig()

    # Initialize pathconf structure
    pathconf.setupTrain(param, 0)

    # Use model dir if given
    if model_dir is not None:
        pathconf.result = model_dir

    # ------------------------------------------------------------------------
    # Run learned network
    start_time = time.clock()
    resize_scale = 1.0
    # If there is not gray image, load the color one and convert to gray
    # read the image
    if os.path.exists(image_file_name.replace(
            "image_color", "image_gray"
    )) and "image_color" in image_file_name:
        image_gray = cv2.imread(image_file_name.replace(
            "image_color", "image_gray"
        ), 0)
        image_color = deepcopy(image_gray)
        image_resized = image_gray
        ratio_h = float(image_resized.shape[0]) / float(image_gray.shape[0])
        ratio_w = float(image_resized.shape[1]) / float(image_gray.shape[1])
    else:
        # read the image
        image_color = cv2.imread(image_file_name)
        image_resized = image_color
        ratio_h = float(image_resized.shape[0]) / float(image_color.shape[0])
        ratio_w = float(image_resized.shape[1]) / float(image_color.shape[1])
        image_gray = cv2.cvtColor(
            image_resized,
            cv2.COLOR_BGR2GRAY).astype(floatX)

    assert len(image_gray.shape) == 2

    end_time = time.clock()
    load_prep_time = (end_time - start_time) * 1000.0
    print("Time taken to read and prepare the image is {} ms".format(
        load_prep_time
    ))

    # check size
    image_height = image_gray.shape[0]
    image_width = image_gray.shape[1]

    # Multiscale Testing
    scl_intv = getattr(param.validation, 'nScaleInterval', 4)
    # min_scale_log2 = 1  # min scale = 2
    # max_scale_log2 = 4  # max scale = 16
    min_scale_log2 = getattr(param.validation, 'min_scale_log2', 1)
    max_scale_log2 = getattr(param.validation, 'max_scale_log2', 4)
    # Test starting with double scale if small image
    min_hw = np.min(image_gray.shape[:2])
    if min_hw <= 1600:
        print("INFO: Testing double scale")
        min_scale_log2 -= 1
    # range of scales to check
    num_division = (max_scale_log2 - min_scale_log2) * (scl_intv + 1) + 1
    scales_to_test = 2**np.linspace(min_scale_log2, max_scale_log2,
                                    num_division)

    # convert scale to image resizes
    resize_to_test = ((float(param.model.nPatchSizeKp - 1) / 2.0) /
                      (param.patch.fRatioScale * scales_to_test))

    # check if resize is valid
    min_hw_after_resize = resize_to_test * np.min(image_gray.shape[:2])
    is_resize_valid = min_hw_after_resize > param.model.nFilterSize + 1

    # if there are invalid scales and resizes
    if not np.prod(is_resize_valid):
        # find first invalid
        first_invalid = np.where(True - is_resize_valid)[0][0]

        # remove scales from testing
        scales_to_test = scales_to_test[:first_invalid]
        resize_to_test = resize_to_test[:first_invalid]

    print('resize to test is {}'.format(resize_to_test))
    print('scales to test is {}'.format(scales_to_test))

    # Run for each scale
    test_res_list = []
    for resize in resize_to_test:

        # Just designate only one scale to bypass resizing. Just a single
        # number is fine, no need for a specific number
        param_cur_scale = deepcopy(param)
        param_cur_scale.patch.fScaleList = [
            1.0
        ]

        # resize according to how we extracted patches when training
        new_height = np.cast['int'](np.round(image_height * resize))
        new_width = np.cast['int'](np.round(image_width * resize))
        start_time = time.clock()
        image = cv2.resize(image_gray, (new_width, new_height))
        end_time = time.clock()
        resize_time = (end_time - start_time) * 1000.0
        print("Time taken to resize image is {}ms".format(
            resize_time
        ))
        total_time += resize_time

        # run test
        if bUseTheano:
            # Load the mean and std
            mean_std_file = pathconf.result + 'mean_std.h5'
            mean_std_dict = loadh5(mean_std_file)
            param_cur_scale.online = paramGroup()
            setattr(param_cur_scale.online, 'mean_x', mean_std_dict['mean_x'])
            setattr(param_cur_scale.online, 'std_x', mean_std_dict['std_x'])

            # disable orientation to avoid complications
            param_cur_scale.model.sOrientation = 'bypass'
            # disable descriptor to avoid complications
            param_cur_scale.model.sDescriptor = 'bypass'

            start_time = time.clock()
            # turn back verbose on
            test_res, compile_time = TestImage(pathconf, param_cur_scale,
                                               image, verbose=False)
            test_res = np.squeeze(test_res)
            end_time = time.clock()
            compute_time = (end_time - start_time) * 1000.0 - compile_time
            print('Time taken using theano for image size {}'
                  ' is {} milliseconds'.format(
                      image.shape, compute_time))
            print("Compile time is {} milliseconds".format(compile_time))

        else:
            start_time = time.clock()
            sKpNonlinearity = getattr(param.model, 'sKpNonlinearity', 'None')
            test_res = apply_learned_filter_2_image_no_theano(
                image, pathconf.result,
                param.model.bNormalizeInput,
                sKpNonlinearity,
                verbose=True)
            end_time = time.clock()
            compute_time = (end_time - start_time) * 1000.0
            print('Time taken using opencv for image size {} is {}'
                  ' milliseconds'.format(image.shape, compute_time))

        total_time += compute_time

        # pad and add to list
        start_time = time.clock()
        test_res_list += [np.pad(test_res,
                                 int((param.model.nFilterSize - 1) / 2),
                                 # mode='edge')]
                                 mode='constant',
                                 constant_values=-np.inf)]
        end_time = time.clock()
        pad_time = (end_time - start_time) * 1000.0
        print("Time taken for padding and stacking is {} ms".format(
            pad_time
        ))
        total_time += pad_time

    # ------------------------------------------------------------------------
    # Non-max suppresion and draw.

    # The nonmax suppression implemented here is very very slow. COnsider this
    # as just a proof of concept implementation as of now.

    # Standard nearby : nonmax will check the approx. the same area as
    # descriptor support region.
    nearby = int(np.round(
        (0.5 * (param.model.nPatchSizeKp - 1.0) *
         float(param.model.nDescInputSize) /
         float(param.patch.nPatchSize))
    ))
    fNearbyRatio = getattr(param.validation, 'fNearbyRatio', 1.0)
    # Multiply by quarter to compensate
    fNearbyRatio *= 0.25
    nearby = int(np.round(nearby * fNearbyRatio))
    nearby = max(nearby, 1)

    nms_intv = getattr(param.validation, 'nNMSInterval', 2)
    edge_th = getattr(param.validation, 'fEdgeThreshold', 10)
    do_interpolation = getattr(param.validation, 'bInterpolate', True)

    # Drawing function, make below True to activate
    bDraw = bSavePng

    def draw_XYZS_to_img(XYZS, file_name):
        """ Drawing functino for displaying """

        if not bDraw:
            return

        # draw onto the original image
        if cv2.__version__[0] == '3':
            linetype = cv2.LINE_AA
        else:
            linetype = cv2.CV_AA
        [cv2.circle(image_color, tuple(np.round(pos).astype(int)),
                    np.round(rad * 6.0).astype(int), (0, 255, 0), 2,
                    lineType=linetype)
         for pos, rad in zip(XYZS[:, :2], XYZS[:, 2])]

        cv2.imwrite(file_name, image_color)

    print("Performing NMS")
    fScaleEdgeness = getattr(param.validation, 'fScaleEdgeness', 0)
    start_time = time.clock()
    res_list = test_res_list
    XYZS = get_XYZS_from_res_list(res_list, resize_to_test,
                                  scales_to_test, nearby, edge_th,
                                  scl_intv, nms_intv, do_interpolation,
                                  fScaleEdgeness)
    end_time = time.clock()
    XYZS = XYZS[:num_keypoint]
    draw_XYZS_to_img(XYZS, output_file + '.jpg')

    nms_time = (end_time - start_time) * 1000.0
    print("NMS time is {} ms".format(nms_time))
    total_time += nms_time
    print("Total time for detection is {} ms".format(total_time))
    if bPrintTime:
        # Also print to a file by appending
        with open("../timing-code/timing.txt", "a") as timing_file:
            print("------ Keypoint Timing ------\n"
                  "NMS time is {} ms\n"
                  "Total time is {} ms\n".format(
                      nms_time, total_time
                  ),
                  file=timing_file)

    if bDraw:
        # resize score to original image size
        res_list = [cv2.resize(score,
                               (image_width, image_height),
                               interpolation=cv2.INTER_NEAREST)
                    for score in test_res_list]
        # make as np array
        res_scores = np.asarray(res_list)
        with h5py.File(output_file + '_scores.h5', 'w') as score_file:
            score_file['score'] = res_scores

    # ------------------------------------------------------------------------
    # Save as keypoint file to be used by the oxford thing
    print("Turning into kp_list")
    kp_list = XYZS2kpList(XYZS)  # note that this is already sorted

    # ------------------------------------------------------------------------
    # Also compute angles with the SIFT method, since the keypoint component
    # alone has no orientations.
    print("Recomputing Orientations")
    new_kp_list, _ = recomputeOrientation(image_gray, kp_list,
                                          bSingleOrientation=True)

    print("Saving to txt")
    saveKpListToTxt(new_kp_list, None, output_file)

#
# compute_detector.py ends here

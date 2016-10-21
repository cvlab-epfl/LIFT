# kp_tools.py ---
#
# Filename: kp_tools.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Sat Feb 20 10:11:28 2016 (+0100)
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

import cv2
import numpy as np
import scipy.ndimage
import six
from scipy.linalg import lu_factor, lu_solve

# Keypoint List Structure Index Info
IDX_X, IDX_Y, IDX_SIZE, IDX_ANGLE, IDX_RESPONSE, IDX_OCTAVE = (
    0, 1, 2, 3, 4, 5)  # , IDX_CLASSID not used
IDX_a, IDX_b, IDX_c = (6, 7, 8)
# NOTE the row-major colon-major adaptation here
IDX_A0, IDX_A2, IDX_A1, IDX_A3 = (9, 10, 11, 12)
# # IDX_CLASSID for KAZE
# IDX_CLASSID = 13

KP_LIST_LEN = 13

# Note that the element in the IDX_SIZE field will be scale, not the opencv
# keypoint's size. As open cv uses diameters, we need to multiply this with two
# to get the opencv size.

# ------------------------------------------------------------------------
# Functions for Swapping between Opencv Kp and Our Kp List
# ------------------------------------------------------------------------


def kp_list_2_opencv_kp_list(kp_list):

    opencv_kp_list = []
    for kp in kp_list:
        opencv_kp = cv2.KeyPoint(x=kp[IDX_X],
                                 y=kp[IDX_Y],
                                 _size=kp[IDX_SIZE] * 2.0,
                                 _angle=kp[IDX_ANGLE],
                                 _response=kp[IDX_RESPONSE],
                                 _octave=np.int32(kp[IDX_OCTAVE]),
                                 # _class_id=np.int32(kp[IDX_CLASSID])
                                 )
        opencv_kp_list += [opencv_kp]

    return opencv_kp_list


def opencv_kp_list_2_kp_list(opencv_kp_list):

    # IMPORTANT: make sure this part corresponds to the one in
    # loadKpListFromTxt

    kp_list = []
    for opencv_kp in opencv_kp_list:
        kp = np.zeros((KP_LIST_LEN, ))
        kp[IDX_X] = opencv_kp.pt[0]
        kp[IDX_Y] = opencv_kp.pt[1]
        kp[IDX_SIZE] = opencv_kp.size * 0.5
        kp[IDX_ANGLE] = opencv_kp.angle
        kp[IDX_RESPONSE] = opencv_kp.response
        kp[IDX_OCTAVE] = opencv_kp.octave
        # Compute a,b,c for vgg affine
        kp[IDX_a] = 1. / (kp[IDX_SIZE]**2)
        kp[IDX_b] = 0.
        kp[IDX_c] = 1. / (kp[IDX_SIZE]**2)
        # Compute A0, A1, A2, A3 and update
        kp = update_affine(kp)
        # kp[IDX_CLASSID] = opencv_kp.class_id

        kp_list += [kp]

    return kp_list


def rescale_circ_kp(kp, fRescale):

    kp[IDX_SIZE] *= fRescale

    return update_affine(update_abc_from_scale(kp))


def update_abc_from_scale(kp):

    kp[IDX_a] = 1. / (kp[IDX_SIZE]**2)
    kp[IDX_b] = 0.
    kp[IDX_c] = 1. / (kp[IDX_SIZE]**2)

    return kp


def update_affine(kp):
    """Returns an updated version of the keypoint.

    Note
    ----
    This function should be applied only to individual keypoints, not a list.

    """

    # Compute A0, A1, A2, A3
    S = np.asarray([[kp[IDX_a], kp[IDX_b]], [kp[IDX_b], kp[IDX_c]]])
    invS = np.linalg.inv(S)
    a = np.sqrt(invS[0, 0])
    b = invS[0, 1] / max(a, 1e-18)
    A = np.asarray([[a, 0], [b, np.sqrt(max(invS[1, 1] - b**2, 0))]])

    # We need to rotate first!
    cos_val = np.cos(np.deg2rad(kp[IDX_ANGLE]))
    sin_val = np.sin(np.deg2rad(kp[IDX_ANGLE]))
    R = np.asarray([[cos_val, -sin_val], [sin_val, cos_val]])

    A = np.dot(A, R)

    kp[IDX_A0] = A[0, 0]
    kp[IDX_A1] = A[0, 1]
    kp[IDX_A2] = A[1, 0]
    kp[IDX_A3] = A[1, 1]

    return kp


# ------------------------------------------------------------------------
# Keypoint I/O operations
# ------------------------------------------------------------------------

def writeFeature2OxFile(file_name, kp_list, desc_list):

    # kp_list : len(kp_list) == number of keypoints, each element is a
    #           opencv Keypoint object
    # desc_list : kp_list.shape == [number of keypoints, dimension],
    # descriptors for each keypoint

    # pt, scale, orientation, desc):
    # write the features in the vlfeat format

    assert len(kp_list) == desc_list.shape[0]

    fp = open(file_name, 'w')  # open the file object

    # write the headings
    # dimension of the descriptor (SIFT is 128)
    fp.write('%d\n' % (desc_list.shape[1]))
    # number of keypoints to be evaluated
    fp.write('%d\n' % (desc_list.shape[0]))

    # write the keypoint information
    for idxKp in six.moves.xrange(len(kp_list)):
        u = kp_list[idxKp][IDX_X]
        v = kp_list[idxKp][IDX_Y]
        a = kp_list[idxKp][IDX_a]
        b = kp_list[idxKp][IDX_b]
        c = kp_list[idxKp][IDX_c]
        desc = desc_list[idxKp, :]

        # turn each kp into vlfeat format
        writestr = '%f %f %f %f %f' % (u, v, a, b,
                                       c) + " " + " ".join(desc.astype(str))

        fp.write(writestr + '\n')

    fp.close()  # close the file object


def loadKpListFromTxt(kp_file_name):

    # Open keypoint file for read
    kp_file = open(kp_file_name, 'rb')

    # skip the first two lines
    kp_line = kp_file.readline()
    kp_line = kp_file.readline()

    kp_list = []
    num_elem = -1
    while True:
        # read a line from file
        kp_line = kp_file.readline()
        # check EOF
        if not kp_line:
            break
        # split read information
        kp_info = kp_line.split()
        parsed_kp_info = []
        for idx in six.moves.xrange(len(kp_info)):
            parsed_kp_info += [float(kp_info[idx])]
        parsed_kp_info = np.asarray(parsed_kp_info)

        if num_elem == -1:
            num_elem = len(parsed_kp_info)
        else:
            assert num_elem == len(parsed_kp_info)

        # IMPORTANT: make sure this part corresponds to the one in
        # opencv_kp_list_2_kp_list

        # check if we have all the kp list info
        if len(parsed_kp_info) == 6:  # if we only have opencv info
            # Compute a,b,c for vgg affine
            a = 1. / (parsed_kp_info[IDX_SIZE]**2)
            b = 0.
            c = 1. / (parsed_kp_info[IDX_SIZE]**2)
            parsed_kp_info = np.concatenate((parsed_kp_info, [a, b, c]))

        if len(parsed_kp_info) == 9:  # if we don't have the Affine warp
            parsed_kp_info = np.concatenate((parsed_kp_info, np.zeros((4, ))))
            parsed_kp_info = update_affine(parsed_kp_info)

        # if len(parsed_kp_info) == 13:
        #     # add dummy class id
        #     parsed_kp_info = np.concatenate((parsed_kp_info, [0]))

        # make sure we have everything!
        assert len(parsed_kp_info) == KP_LIST_LEN

        kp_list += [parsed_kp_info]

    # Close keypoint file
    kp_file.close()

    return kp_list


def saveKpListToTxt(kp_list, orig_kp_file_name, kp_file_name):

    # first line KP_LIST_LEN to indicate we have the full
    kp_line = str(KP_LIST_LEN) + '\n'

    # Open keypoint file for write
    kp_file = open(kp_file_name, 'w')

    # write the first line
    kp_file.write(kp_line)

    # write the number of kp in second line
    kp_file.write('{}\n'.format(len(kp_list)))

    for kp in kp_list:

        # Make sure we have all info for kp
        assert len(kp) == KP_LIST_LEN

        # Form the string to write
        write_string = ""
        for kp_elem, _i in zip(kp, range(len(kp))):
            # if _i == IDX_OCTAVE or _i == IDX_CLASSID:
            if _i == IDX_OCTAVE:  # in case of the octave
                write_string += str(np.int32(kp_elem)) + " "
            else:
                write_string += str(kp_elem) + " "
        write_string += "\n"

        # Write the string
        kp_file.write(write_string)

    # Close keypoint file
    kp_file.close()


def XYZS2kpList(XYZS):

    kp_list = [None] * XYZS.shape[0]
    for idxKp in six.moves.xrange(XYZS.shape[0]):

        kp = np.zeros((KP_LIST_LEN, ))
        kp[IDX_X] = XYZS[idxKp, 0]
        kp[IDX_Y] = XYZS[idxKp, 1]
        kp[IDX_SIZE] = XYZS[idxKp, 2]
        kp[IDX_ANGLE] = 0
        kp[IDX_RESPONSE] = XYZS[idxKp, 3]

        # computing the octave should be dealt with caution. We compute in the
        # SIFT way. The SIFT code of openCV computes scale in the following
        # way.
        # >>> scale = 1.6 * 2**((layer+xi) / 3) * 2**octave
        # where octave is packed by octave = layer << 8 + octv
        layer_n_octv = np.log2(kp[IDX_SIZE] / 1.6)
        layer_n_octv = max(0, layer_n_octv)  # TODO: FOR NOW LET'S JUST DO THIS
        octv = int(np.floor(layer_n_octv))
        layer_n_xi = (layer_n_octv - np.floor(layer_n_octv)) * 3.0
        layer = int(np.floor(layer_n_xi))
        xi = layer_n_xi - np.floor(layer_n_xi)
        # make sure we have the layer correctly by checking that it is smaller
        # than 3
        assert layer < 3
        # pack octave in the same way as openCV
        # kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
        # also remember
        # octave = octave < 128 ? octave : (-128 | octave);
        # which means
        # kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255)
        # so if octave is negative & 255 will give otcv >= 128 later...
        octv = octv & 255
        octave = octv + (layer << 8) + (int(np.round((xi + 0.5) * 255.)) << 16)
        kp[IDX_OCTAVE] = octave

        # Compute a,b,c for vgg affine
        kp[IDX_a] = 1. / (kp[IDX_SIZE]**2)
        kp[IDX_b] = 0.
        kp[IDX_c] = 1. / (kp[IDX_SIZE]**2)
        # Compute A0, A1, A2, A3 and update
        kp = update_affine(kp)
        # kp[IDX_CLASSID] = 0

        kp_list[idxKp] = kp

    return kp_list


# ------------------------------------------------------------------------
# NonMaxSuppresion related
# ------------------------------------------------------------------------


def get_XYZS_from_res_list(res_list, resize_to_test, scales_to_test, nearby=1,
                           edge_th=0, scl_intv=2, nms_intv=1,
                           do_interpolation=False, fScaleEdgeness=0.0):
    # NMS
    nms_res = nonMaxSuppression(res_list, nearby=nearby,
                                scl_intv=scl_intv, nms_intv=nms_intv)

    XYZS = get_subpixel_XYZS(res_list, nms_res, resize_to_test,
                             scales_to_test, edge_th, do_interpolation,
                             fScaleEdgeness)
    # sort by score
    XYZS = XYZS[np.argsort(XYZS[:, 3])[::-1]]

    return XYZS


def get_subpixel_XYZS(score_list, nms_list, resize_to_test,
                      scales_to_test, edge_th, do_interpolation,
                      fScaleEdgeness):

    log_scales = np.log2(scales_to_test)
    log_scale_step = ((np.max(log_scales) - np.min(log_scales)) /
                      (len(scales_to_test) - 1.0))

    X = [()] * len(nms_list)
    Y = [()] * len(nms_list)
    Z = [()] * len(nms_list)
    S = [()] * len(nms_list)
    for idxScale in six.moves.xrange(len(nms_list)):
        nms = nms_list[idxScale]

        pts = np.where(nms)
        if len(pts[0]) == 0:
            continue

        assert idxScale > 0 and idxScale < len(nms_list) - 1

        # compute ratio for coordinate conversion
        fRatioUp = (
            (np.asarray(score_list[idxScale + 1].shape, dtype='float') - 1.0) /
            (np.asarray(score_list[idxScale].shape, dtype='float') - 1.0)
        ).reshape([2, -1])
        fRatioDown = (
            (np.asarray(score_list[idxScale - 1].shape, dtype='float') - 1.0) /
            (np.asarray(score_list[idxScale].shape, dtype='float') - 1.0)
        ).reshape([2, -1])

        # the conversion function
        def at(dx, dy, ds):
            if not isinstance(dx, np.ndarray):
                dx = np.ones(len(pts[0]),) * dx
            if not isinstance(dy, np.ndarray):
                dy = np.ones(len(pts[0]),) * dy
            if not isinstance(ds, np.ndarray):
                ds = np.ones(len(pts[0]),) * ds
            new_pts = (pts[0] + dy, pts[1] + dx)
            ds = np.round(ds).astype(int)
            fRatio = ((ds == 0).reshape([1, -1]) * 1.0 +
                      (ds == -1).reshape([1, -1]) * fRatioDown +
                      (ds == 1).reshape([1, -1]) * fRatioUp)
            assert np.max(ds) <= 1 and np.min(ds) >= -1
            new_pts = tuple([np.round(v * r).astype(int)
                             for v, r in zip(new_pts, fRatio)])
            scores_to_return = np.asarray([
                score_list[idxScale + _ds][_y, _x]
                for _ds, _x, _y in zip(
                    ds, new_pts[1], new_pts[0]
                )
            ])
            return scores_to_return

        # compute the gradient
        Dx = 0.5 * (at(+1, 0, 0) - at(-1, 0, 0))
        Dy = 0.5 * (at(0, +1, 0) - at(0, -1, 0))
        Ds = 0.5 * (at(0, 0, +1) - at(0, 0, -1))

        # compute the Hessian
        Dxx = (at(+1, 0, 0) + at(-1, 0, 0) - 2.0 * at(0, 0, 0))
        Dyy = (at(0, +1, 0) + at(0, -1, 0) - 2.0 * at(0, 0, 0))
        Dss = (at(0, 0, +1) + at(0, 0, -1) - 2.0 * at(0, 0, 0))

        Dxy = 0.25 * (at(+1, +1, 0) + at(-1, -1, 0) -
                      at(-1, +1, 0) - at(+1, -1, 0))
        Dxs = 0.25 * (at(+1, 0, +1) + at(-1, 0, -1) -
                      at(-1, 0, +1) - at(+1, 0, -1))
        Dys = 0.25 * (at(0, +1, +1) + at(0, -1, -1) -
                      at(0, -1, +1) - at(0, +1, -1))

        # filter out all keypoints which we have inf
        is_good = (np.isfinite(Dx) * np.isfinite(Dy) * np.isfinite(Ds) *
                   np.isfinite(Dxx) * np.isfinite(Dyy) * np.isfinite(Dss) *
                   np.isfinite(Dxy) * np.isfinite(Dxs) * np.isfinite(Dys))
        Dx = Dx[is_good]
        Dy = Dy[is_good]
        Ds = Ds[is_good]
        Dxx = Dxx[is_good]
        Dyy = Dyy[is_good]
        Dss = Dss[is_good]
        Dxy = Dxy[is_good]
        Dxs = Dxs[is_good]
        Dys = Dys[is_good]
        pts = tuple([v[is_good] for v in pts])
        # check if empty
        if len(pts[0]) == 0:
            continue

        # filter out all keypoints which are on edges
        if edge_th > 0:

            # # re-compute the Hessian
            # Dxx = (at(b[:, 0] + 1, b[:, 1], b[:, 2]) +
            #        at(b[:, 0] - 1, b[:, 1], b[:, 2]) -
            #        2.0 * at(b[:, 0], b[:, 1], b[:, 2]))
            # Dyy = (at(b[:, 0], b[:, 1] + 1, b[:, 2]) +
            #        at(b[:, 0], b[:, 1] - 1, b[:, 2]) -
            #        2.0 * at(b[:, 0], b[:, 1], b[:, 2]))

            # Dxy = 0.25 * (at(b[:, 0] + 1, b[:, 1] + 1, b[:, 2]) +
            #               at(b[:, 0] - 1, b[:, 1] - 1, b[:, 2]) -
            #               at(b[:, 0] - 1, b[:, 1] + 1, b[:, 2]) -
            #               at(b[:, 0] + 1, b[:, 1] - 1, b[:, 2]))

            # H = np.asarray([[Dxx, Dxy, Dxs],
            #                 [Dxy, Dyy, Dys],
            #                 [Dxs, Dys, Dss]]).transpose([2, 0, 1])

            edge_score = (Dxx + Dyy) * (Dxx + Dyy) / (Dxx * Dyy - Dxy * Dxy)
            is_good = ((edge_score >= 0) *
                       (edge_score < (edge_th + 1.0)**2 / edge_th))

            if fScaleEdgeness > 0:
                is_good = is_good * (
                    abs(Dss) > fScaleEdgeness
                )

            Dx = Dx[is_good]
            Dy = Dy[is_good]
            Ds = Ds[is_good]
            Dxx = Dxx[is_good]
            Dyy = Dyy[is_good]
            Dss = Dss[is_good]
            Dxy = Dxy[is_good]
            Dxs = Dxs[is_good]
            Dys = Dys[is_good]
            pts = tuple([v[is_good] for v in pts])
            # check if empty
            if len(pts[0]) == 0:
                continue

        b = np.zeros((len(pts[0]), 3))
        if do_interpolation:
            # from VLFEAT

            # solve linear system
            A = np.asarray([[Dxx, Dxy, Dxs],
                            [Dxy, Dyy, Dys],
                            [Dxs, Dys, Dss]]).transpose([2, 0, 1])

            b = np.asarray([-Dx, -Dy, -Ds]).transpose([1, 0])

            b_solved = np.zeros_like(b)
            for idxPt in six.moves.xrange(len(A)):
                b_solved[idxPt] = lu_solve(lu_factor(A[idxPt]), b[idxPt])

            b = b_solved

        # throw away the ones with bad subpixel localizatino
        is_good = ((abs(b[:, 0]) < 1.5) * (abs(b[:, 1]) < 1.5) *
                   (abs(b[:, 2]) < 1.5))
        b = b[is_good]
        pts = tuple([v[is_good] for v in pts])
        # check if empty
        if len(pts[0]) == 0:
            continue

        x = pts[1] + b[:, 0]
        y = pts[0] + b[:, 1]
        log_ds = b[:, 2]

        S[idxScale] = at(b[:, 0], b[:, 1], b[:, 2])
        X[idxScale] = x / resize_to_test[idxScale]
        Y[idxScale] = y / resize_to_test[idxScale]
        Z[idxScale] = scales_to_test[idxScale] * 2.0**(log_ds * log_scale_step)

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    Z = np.concatenate(Z)
    S = np.concatenate(S)

    XYZS = np.concatenate([X.reshape([-1, 1]),
                           Y.reshape([-1, 1]),
                           Z.reshape([-1, 1]),
                           S.reshape([-1, 1])],
                          axis=1)

    return XYZS


def nonMaxSuppression(score_img_or_list, nearby=1, scl_intv=2, nms_intv=1):
    """ Performs Non Maximum Suppression.

    Parameters
    ----------
    score_img_or_list: nparray or list
        WRITEME

    nearby: int
        Size of the neighborhood.

    scl_intv: int
        How many levels we have between half scale.

    nms_intv: int
        How many levels we use for scale space nms.

    """

    filter_size = (nearby * 2 + 1,) * 2

    if isinstance(score_img_or_list, list):
        bis2d = False
    else:
        bis2d = True

    if bis2d:
        score = score_img_or_list
        # max score in region
        max_score = scipy.ndimage.filters.maximum_filter(
            score, filter_size, mode='constant', cval=-np.inf
        )
        # second score in region
        second_score = scipy.ndimage.filters.rank_filter(
            score, -2, filter_size, mode='constant', cval=-np.inf
        )
        # min score in region to check infs
        min_score = scipy.ndimage.filters.minimum_filter(
            score, filter_size, mode='constant', cval=-np.inf
        )
        nonmax_mask_or_list = ((score == max_score) *
                               (max_score > second_score) *
                               np.isfinite(min_score))

    else:

        max2d_list = [
            scipy.ndimage.filters.maximum_filter(
                score, filter_size, mode='constant', cval=-np.inf
            )
            for score in score_img_or_list
        ]

        second2d_list = [
            scipy.ndimage.filters.rank_filter(
                score, -2, filter_size, mode='constant', cval=-np.inf
            )
            for score in score_img_or_list
        ]

        min2d_list = [
            scipy.ndimage.filters.minimum_filter(
                score, filter_size, mode='constant', cval=-np.inf
            )
            for score in score_img_or_list
        ]

        nonmax2d_list = [(score == max_score) * (max_score > second_score) *
                         np.isfinite(min_score)
                         for score, max_score, second_score, min_score in
                         zip(score_img_or_list,
                             max2d_list,
                             second2d_list,
                             min2d_list)
                         ]

        nonmax_mask_or_list = [None] * len(nonmax2d_list)
        for idxScale in six.moves.xrange(len(nonmax2d_list)):

            nonmax2d = nonmax2d_list[idxScale]
            max2d = max2d_list[idxScale]

            # prep output
            nonmax_mask = np.zeros_like(nonmax2d)

            # get coordinates of the local max positions of nonmax2d
            coord2d_max = np.where(nonmax2d)
            # print("2d nonmax = {} kp".format(len(coord2d_max[0])))

            # range of other scales to look at
            # scl_diffs = np.arange(-scl_intv, scl_intv + 1)
            # scl_diffs = np.arange(-1, 1 + 1)
            scl_diffs = np.arange(-nms_intv, nms_intv + 1)
            scl_diffs = scl_diffs[scl_diffs != 0]

            # skip if we don't have the complete set
            if (idxScale + np.min(scl_diffs) < 0 or
                    idxScale + np.max(scl_diffs) > len(nonmax2d_list) - 1):
                continue

            # Test on multiple scales to see if it is scalespace local max
            for scl_diff in scl_diffs:

                scl_to_compare = idxScale + scl_diff

                # look at the other scales max
                max2d_other = max2d_list[scl_to_compare]
                # compute ratio for coordinate conversion
                fRatio \
                    = (np.asarray(max2d_other.shape, dtype='float') - 1.0) \
                    / (np.asarray(nonmax2d.shape, dtype='float') - 1.0)
                # get indices for lower layer
                coord_other = tuple([np.round(v * r).astype(int)
                                     for v, r in zip(coord2d_max, fRatio)])
                # find good points which should survive
                idxGood = np.where((max2d[coord2d_max] >
                                    max2d_other[coord_other]) *
                                   np.isfinite(max2d_other[coord_other])
                                   )

                # copy only the ones that are good
                coord2d_max = tuple([v[idxGood] for v in coord2d_max])

            # mark surviving points
            nonmax_mask[coord2d_max] = 1.0

            # special case when we are asked with single item in list
            if len(nonmax2d_list) == 1:
                # get coordinates of the local max positions of nonmax2d
                coord2d_max = np.where(nonmax2d)
                # mark surviving points
                nonmax_mask[coord2d_max] = 1.0

            # add to list
            nonmax_mask_or_list[idxScale] = nonmax_mask

    return nonmax_mask_or_list


#
# kp_tools.py ends here

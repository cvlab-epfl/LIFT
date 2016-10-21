# helper.py ---
#
# Filename: helper.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Tue Feb 23 15:18:50 2016 (+0100)
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
import six


def create_perturb(orig_pos, nPatchSize, nDescInputSize, fPerturbInfo):
    """Create perturbations in the xyz format we use in the keypoint component.

    The return value is in x,y,z where x and y are scaled coordinates that are
    between -1 and 1. In fact, the coordinates are scaled so that 1 = 0.5 *
    (nPatchSize-1) + patch center.

    In case of the z, it is in log2 scale, where zero corresponds to the
    original scale of the keypoint. For example, z of 1 would correspond to 2
    times the original scale of the keypoint.

    Parameters
    ----------
    orig_pos: ndarray of float
        Center position of the patch in coordinates.

    nPatchSize: int
        Patch size of the entire patch extraction region.

    nDescInputSize: int
        Patch size of the descriptor support region.

    fPerturbInfo: ndarray
        Amount of maximum perturbations we allow in xyz.

    Notes
    -----
    The perturbation has to ensure that the *ground truth* support region in
    included.
    """

    # Check provided perturb info so that it is guaranteed that it will
    # generate perturbations that holds the ground truth support region
    assert fPerturbInfo[0] <= 1.0 - float(nDescInputSize) / float(nPatchSize)
    assert fPerturbInfo[1] <= 1.0 - float(nDescInputSize) / float(nPatchSize)

    # Generate random perturbations
    perturb_xyz = ((2.0 * np.random.rand(orig_pos.shape[0], 3) - 1.0) *
                   fPerturbInfo.reshape([1, 3]))

    return perturb_xyz


def apply_perturb(orig_pos, perturb_xyz, maxRatioScale):
    """Apply the perturbation to ``a'' keypoint.

    The xyz perturbation is in format we use in the keypoint component. See
    'create_perturb' for details.

    Parameters
    ----------
    orig_pos: ndarray of float (ndim == 1, size 3)
        Original position of a *single* keypoint in pixel coordinates.

    perturb_xyz: ndarray of float (ndim == 1, size 3)
        The center position in xyz after the perturbation is applied.

    maxRatioScale: float
        The multiplier that get's multiplied to scale to get half-width of the
        region we are going to crop.
    """

    # assert that we are working with only one sample
    assert len(orig_pos.shape) == 1
    assert len(perturb_xyz.shape) == 1

    # get the new scale
    new_pos_s = orig_pos[2] * (2.0**(-perturb_xyz[2]))

    # get xy to pixels conversion
    xyz_to_pix = new_pos_s * maxRatioScale

    # Get the new x and y according to scale. Note that we multiply the
    # movement we need to take by 2.0**perturb_xyz since we are moving at a
    # different scale
    new_pos_x = orig_pos[0] - perturb_xyz[0] * 2.0**perturb_xyz[2] * xyz_to_pix
    new_pos_y = orig_pos[1] - perturb_xyz[1] * 2.0**perturb_xyz[2] * xyz_to_pix

    perturbed_pos = np.asarray([new_pos_x, new_pos_y, new_pos_s])

    return perturbed_pos


def get_crop_range(xx, yy, half_width):
    """Function for retrieving the crop coordinates"""

    xs = np.cast['int'](np.round(xx - half_width))
    xe = np.cast['int'](np.round(xx + half_width))
    ys = np.cast['int'](np.round(yy - half_width))
    ye = np.cast['int'](np.round(yy + half_width))

    return xs, xe, ys, ye


def crop_patch(img, cx, cy, clockwise_rot, resize_ratio, nPatchSize):
    """Crops the patch.

    Crops with center at cx, cy with patch size of half_width and resizes to
    nPatchSize x nPatchsize patch.

    Parameters
    ----------
    img: np.ndarray
        Input image to be cropped from.

    cx: float
        x coordinate of the patch.

    cy: float
        y coordinate of the patch.

    clockwise_rot: float (degrees)
        clockwise rotation to apply when extracting

    resize_ratio: float
        Ratio of the resize. For example, ratio of two will crop a 2 x
        nPatchSize region.

    nPatchSize: int
        Size of the returned patch.

    Notes
    -----

    The cv2.warpAffine behaves in similar way to the spatial transformers. The
    M matrix should move coordinates from the original image to the patch,
    i.e. inverse transformation.

    """

    # Below equation should give (nPatchSize-1)/2 when M x [cx, 0, 1]',
    # 0 when M x [cx - (nPatchSize-1)/2*resize_ratio, 0, 1]', and finally,
    # nPatchSize-1 when M x [cx + (nPatchSize-1)/2*resize_ratio, 0, 1]'.
    dx = (nPatchSize - 1.0) * 0.5 - cx / resize_ratio
    dy = (nPatchSize - 1.0) * 0.5 - cy / resize_ratio
    M = np.asarray([[1. / resize_ratio, 0.0, dx],
                    [0.0, 1. / resize_ratio, dy],
                    [0.0, 0.0, 1.0]])
    # move to zero base before rotation
    R_pre = np.asarray([[1.0, 0.0, -(nPatchSize - 1.0) * 0.5],
                        [0.0, 1.0, -(nPatchSize - 1.0) * 0.5],
                        [0.0, 0.0, 1.0]])
    # rotate
    theta = clockwise_rot / 180.0 * np.pi
    R_rot = np.asarray([[np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0]])
    # move back to corner base
    R_post = np.asarray([[1.0, 0.0, (nPatchSize - 1.0) * 0.5],
                         [0.0, 1.0, (nPatchSize - 1.0) * 0.5],
                         [0.0, 0.0, 1.0]])
    # combine
    R = np.dot(R_post, np.dot(R_rot, R_pre))

    crop = cv2.warpAffine(img, np.dot(R, M)[:2, :], (nPatchSize, nPatchSize))

    return crop


def load_patches(img, kp_in, y_in, ID_in, angle_in, fRatioScale, fMaxScale,
                 nPatchSize, nDescInputSize, in_dim, bPerturb, fPerturbInfo,
                 bReturnCoords=False, nAugmentedRotations=1,
                 fAugmentRange=180.0, fAugmentCenterRandStrength=0.0,
                 sAugmentCenterRandMethod="uniform"):
    '''Loads Patches given img and list of keypoints

    Parameters
    ----------

    img: input grayscale image (or color)

    kp_in: 2-D array of keypoints [nbkp, 3+], Note that data is in order of
        [x,y,scale,...]

    y_in: 1-D array of keypoint labels (or target values)

    ID_in: 1-D array of keypoint IDs

    fRatioScale: float
        The ratio which gets multiplied to obtain the crop radius. For example
        if fRatioScale is (48-1) /2 /2, and the scale is 2, the crop region
        will be of size 48x48. Note that -1 is necessary as the center pixel is
        inclusive. In our previous implementation regarding ECCV submission, -1
        was negelected. This however should not make a bit difference.

    fMaxScale: float
        Since we are trying to make scale space learning possible, we give list
        of scales for training. This should contain the maximum value, so that
        we consider the largest scale when cropping.

    nPatchSize: int
        Size of the patch (big one, not to be confused with nPatchSizeKp).

    nDescInputSize: int
        Size of the inner patch (descriptor support region). Used for computing
        the bounds for purturbing the patch location.

    in_dim: int
        Number of dimensions of the input. For example grayscale would mean
        `in_dim == 1`.

    bPerturb: boolean
        Whether to perturb when extracting patches

    fPerturbInfo: np.array (float)
        How much perturbation (in relative scale) for x, y, scale

    bReturnCoord: boolean
        Return groundtruth coordinates. Should be set to True for new
        implementations. Default is False for backward compatibility.

    nAugmentedRotations: int
        Number of augmented rotations (equaly spaced) to be added to the
        dataset. The original implementation should be equal to having this
        number set to 1.

    fAugmentRange: float (degrees)
        The range of the augnmented degree. For example, 180 would mean the
        full range, 90 would be the half range.

    fAugmentCenterRandStrength: float (degrees)
        The strength of the random to be applied to the center angle
        perturbation.

    sAugmentCenterRandMethod: string
        Name of the center randomness method.

    '''

    # get max possible scale ratio
    maxRatioScale = fRatioScale * fMaxScale

    # check validity of  nPreRotPatchSize
    assert nAugmentedRotations >= 1
    # # Since we apply perturbations, we need to be at least sqrt(2) larger than
    # # the desired when random augmentations are introduced
    # if nAugmentedRotations > 1 or fAugmentCenterRandStrength > 0:
    #     nInitPatchSize = np.round(np.sqrt(2.0) * nPatchSize).astype(int)
    # else:
    #     nInitPatchSize = nPatchSize

    # pre-allocate maximum possible array size for data
    x = np.zeros((kp_in.shape[0] * nAugmentedRotations, in_dim,
                  nPatchSize, nPatchSize), dtype='uint8')
    y = np.zeros((kp_in.shape[0] * nAugmentedRotations,), dtype='float32')
    ID = np.zeros((kp_in.shape[0] * nAugmentedRotations,), dtype='int')
    pos = np.zeros((kp_in.shape[0] * nAugmentedRotations, 3), dtype='float')
    angle = np.zeros((kp_in.shape[0] * nAugmentedRotations,), dtype='float32')
    coords = np.tile(np.zeros_like(kp_in), (nAugmentedRotations, 1))

    # create perturbations
    # Note: the purturbation still considers only the nPatchSize
    perturb_xyz = create_perturb(kp_in, nPatchSize,
                                 nDescInputSize, fPerturbInfo)

    # delete perturbations for the negatives (look at kp[6])
    # perturb_xyz[kp_in[:, 6] == 0] = 0
    perturb_xyz[y_in == 0] = 0

    idxKeep = 0
    for idx in six.moves.xrange(kp_in.shape[0]):

        # current kp position
        cur_pos = apply_perturb(kp_in[idx], perturb_xyz[idx],  maxRatioScale)
        cx = cur_pos[0]
        cy = cur_pos[1]
        cs = cur_pos[2]

        # retrieve the half width acording to scale
        max_hw = cs * maxRatioScale

        # get crop range considering bigger area
        xs, xe, ys, ye = get_crop_range(cx, cy, max_hw * np.sqrt(2.0))

        # boundary check with safety margin
        safety_margin = 1
        # if xs < 0 or xe >= img.shape[1] or ys < 0 or ys >= img.shape[0]:
        if (xs < safety_margin or xe >= img.shape[1] - safety_margin or
                ys < safety_margin or ys >= img.shape[0] - safety_margin):
            continue

        # create an initial center orientation
        center_rand = 0
        if sAugmentCenterRandMethod == "uniform":
            # Note that the below will give zero when
            # `fAugmentCenterRandStrength == 0`. This effectively disables the
            # random perturbation.
            center_rand = ((np.random.rand() * 2.0 - 1.0) *
                           fAugmentCenterRandStrength)
        else:
            raise NotImplementedError(
                "Unknown random method "
                "sAugmentCenterRandMethod = {}".format(
                    sAugmentCenterRandMethod
                )
            )

        # create an array of rotations to used
        rot_diff_list = np.arange(nAugmentedRotations).astype(float)
        # compute the step between subsequent rotations
        rot_step = 2.0 * fAugmentRange / float(nAugmentedRotations)
        rot_diff_list *= rot_step

        for rot_diff in rot_diff_list:

            # the rotation to be applied for this patch
            crot_deg = rot_diff + center_rand
            crot_rad = crot_deg * np.pi / 180.0

            # Crop using the crop function
            # crop = img[ys:ye, xs:xe]
            # x[idxKeep, 0, :, :] = crop_patch(
            #     img, cx, cy, crot_deg,
            #     max_hw / (float(nPatchSize - 1) * 0.5),
            #     nPatchSize)     # note the nInitPatchSize
            cur_patch = crop_patch(
                img, cx, cy, crot_deg,
                max_hw / (float(nPatchSize - 1) * 0.5),
                nPatchSize)
            if len(cur_patch.shape) == 2:
                cur_patch = cur_patch[..., np.newaxis]
            x[idxKeep] = cur_patch.transpose(2, 0, 1)

            # crop = img[ys:ye, xs:xe]

            # # resize to patch (TODO: change 0 here so that we can deal with
            # # not gray stuff)
            # x[idxKeep, 0, :, :] = cv2.resize(
            #     crop, (nPatchSize, nPatchSize)).copy()

            # update target value and id
            y[idxKeep] = y_in[idx]
            ID[idxKeep] = ID_in[idx]
            # add crot (in radians), note that we are between -2pi and 0 for
            # compatiblity
            angle[idxKeep] = ((angle_in[idx] + crot_rad) % (2.0 * np.pi) -
                              (2.0 * np.pi))
            # raise NotImplementedError(
            #     "TODO: Make sure we alter this properly "
            #     "-- used for bypass modules"
            # )

            # Store the perturbation (center of the patch is 0,0,0)
            new_perturb_xyz = perturb_xyz[idx].copy()
            xx, yy, zz = new_perturb_xyz
            rx = np.cos(crot_rad) * xx - np.sin(crot_rad) * yy
            ry = np.sin(crot_rad) * xx + np.cos(crot_rad) * yy
            rz = zz
            pos[idxKeep] = np.asarray([rx, ry, rz])
            # raise NotImplementedError(
            #     "TODO: Make sure we alter this properly"
            # )

            # store the original pixel coordinates
            new_kp_in = kp_in[idx].copy()
            new_kp_in[3] = ((new_kp_in[3] + crot_rad) % (2.0 * np.pi) -
                            (2.0 * np.pi))
            coords[idxKeep] = new_kp_in
            # raise NotImplementedError(
            #     "TODO: Make sure we alter this properly -- used for testing"
            # )

            idxKeep += 1

            # raise NotImplementedError(
            #     "TODO: end of the nAugmentedRotations loop"
            # )

    # Delete unassigned
    x = x[:idxKeep]
    y = y[:idxKeep]
    ID = ID[:idxKeep]
    pos = pos[:idxKeep]
    angle = angle[:idxKeep]
    coords = coords[:idxKeep]

    if not bReturnCoords:
        return x.astype("uint8"), y, ID.astype("int"), pos, angle
    else:
        return x.astype("uint8"), y, ID.astype("int"), pos, angle, coords


#
# helper.py ends here

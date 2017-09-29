#!/bin/bash

# ------------------------------------------------------------
# Example script for running LIFT

# Open MP Settings
export OMP_NUM_THREADS=1

# Cuda Settings
export CUDA_VISIBLE_DEVICES=0

# Theano Flags 
export THEANO_FLAGS="device=gpu0,${THEANO_FLAGS}"

# ------------------------------------------------------------
# LIFT code settings

# Number of keypoints
_LIFT_NUM_KEYPOINT=1000

# Whether to save debug image for keypoints
_LIFT_SAVE_PNG=1

# Whether the use Theano when keypoint testing. CuDNN is required when turned
# on
_LIFT_USE_THEANO=1

# The base path of the code
export _LIFT_BASE_PATH="$(pwd)"

_LIFT_PYTHON_CODE_PATH="${_LIFT_BASE_PATH}/python-code"

_LIFT_C_CODE_PATH="${_LIFT_BASE_PATH}/c-code"

# Check if cmake is installed
_CMAKE_PATH=`which cmake`
if [ ! -f "${_CMAKE_PATH}" ]
then
    echo "CMAKE is not installed!"
    exit 
fi

# Make sure libSIFT is compiled
if [ ! -f "${_LIFT_C_CODE_PATH}/libSIFT.so" ]
then
    (cd "${_LIFT_C_CODE_PATH}/build"; \
     cmake .. && make
    )
fi

# Test image and model settings
_LIFT_TEST_IMG_NAME="img1"
_LIFT_TEST_IMG="${_LIFT_BASE_PATH}/data/testimg/${_LIFT_TEST_IMG_NAME}.jpg"
_LIFT_TEST_CONFIG="${_LIFT_BASE_PATH}/models/configs/picc-finetune-nopair.config"
_LIFT_MODEL_DIR="${_LIFT_BASE_PATH}/models/picc-best/"

# Output Settings
_LIFT_RES_DIR="${_LIFT_BASE_PATH}/results"
_LIFT_KP_FILE_NAME="${_LIFT_RES_DIR}/${_LIFT_TEST_IMG_NAME}_kp.txt"
_LIFT_ORI_FILE_NAME="${_LIFT_RES_DIR}/${_LIFT_TEST_IMG_NAME}_ori.txt"
_LIFT_DESC_FILE_NAME="${_LIFT_RES_DIR}/${_LIFT_TEST_IMG_NAME}_desc.h5"


(cd $_LIFT_PYTHON_CODE_PATH; \
 python compute_detector.py \
	$_LIFT_TEST_CONFIG \
	$_LIFT_TEST_IMG \
	$_LIFT_KP_FILE_NAME \
	$_LIFT_SAVE_PNG \
	$_LIFT_USE_THEANO \
	0 \
	$_LIFT_MODEL_DIR \
	$_LIFT_NUM_KEYPOINT \
 )

(cd $_LIFT_PYTHON_CODE_PATH; \
 python compute_orientation.py \
	$_LIFT_TEST_CONFIG \
	$_LIFT_TEST_IMG \
	$_LIFT_KP_FILE_NAME \
	$_LIFT_ORI_FILE_NAME \
	0 \
	0 \
	$_LIFT_MODEL_DIR
)

(cd $_LIFT_PYTHON_CODE_PATH; \
 python compute_descriptor.py \
	$_LIFT_TEST_CONFIG \
	$_LIFT_TEST_IMG \
	$_LIFT_ORI_FILE_NAME \
	$_LIFT_DESC_FILE_NAME \
	0 \
	0 \
	$_LIFT_MODEL_DIR
)


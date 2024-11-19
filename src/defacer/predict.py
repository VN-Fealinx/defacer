#!/usr/bin/env python

# # predict something from one multi-modal nifti images
# Tested with Python 3.7, Tensorflow 2.7
# @author : Philippe Boutinaud - Fealinx

import gc
import os
import time
import json

import argparse
from pathlib import Path
import hashlib

import numpy as np
import nibabel
import tensorflow as tf


def md5(fname):
    """
    Create a md5 hash for a file or a folder

    Args:
        fname (str): file/folder path

    Returns:
        str: hexadecimal hash for the file/folder
    """
    hash_md5 = hashlib.md5()
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    elif os.path.isdir(fname):
        fpath = Path(fname)
        file_list = [f for f in fpath.rglob('*') if os.path.isfile(f)]
        file_list.sort()
        for sub_file in file_list:
            with open(sub_file, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
    else:
        raise FileNotFoundError(f'The input is neither a file nor a folder: {fname}')
    return hash_md5.hexdigest()


def _load_image(filename):
    data_nii = nibabel.load(filename)
    # load file and add dimension for the modality
    image = data_nii.get_fdata(dtype=np.float32)[..., np.newaxis]
    return image, data_nii.affine


# Script parameters
def predict_parser():
    parser = argparse.ArgumentParser(
        description="Run inference with tensorflow models(s) on an image that may be built from several modalities"
    )
    parser.add_argument(
        "--img",
        type=Path,
        action='store',
        help="Input image",
        required=True)

    parser.add_argument(
        "-m", "--model",
        type=Path,
        action='store',
        help="(multiple) input modality",
        required=True)

    parser.add_argument(
        "-descriptor",
        type=Path,
        required=True,
        help="File info json about model")

    parser.add_argument(
        "-b", "--braimask",
        type=Path,
        help="brain mask image")

    parser.add_argument(
        "-o", "--out_dir",
        type=Path,
        help="path for the output file (output of the inference from tensorflow model)")

    parser.add_argument(
        "-g", "--gpu",
        type=int,
        help="GPU card ID; for CPU use -1")

    parser.add_argument(
        "--threads",
        default=1,
        type=int,
        help="Number of threads to use when running on CPU"
    )

    parser.add_argument(
        "--verbose",
        help="increase output verbosity",
        action="store_true")

    return parser


def main():
    pred_parser = predict_parser()
    args = pred_parser.parse_args()

    _VERBOSE = args.verbose

    # Set GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        if args.gpu < 0:
            tf.config.set_visible_devices([], 'GPU')
            tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        if _VERBOSE:
            if args.gpu >= 0:
                print(f"Trying to run inference on GPU {args.gpu}")
            else:
                print("Trying to run inference on CPU")
    else:
        print(f"Trying to run inference on GPU {tf.config.get_visible_devices('GPU')}")

    # The tf model files for the predictors, the prediction will be averaged
    predictor_files = []
    path = args.descriptor
    with open(path) as f:
        meta_data = json.load(f)

    for mfile in meta_data['files']:
        mdirname = os.path.basename(args.model)
        mfilename = mfile['name']
        model_file = os.path.join(args.model, mfilename)
        if mfilename[:len(mdirname)] == mdirname and not os.path.exists(model_file):  # model dir is in both args.model and mfile['name']
            mfilename = os.path.join(*mfilename.split(os.sep)[1:])
            model_file = os.path.join(args.model, mfilename)
        predictor_files.append(model_file)

    if len(predictor_files) == 0:
        raise FileNotFoundError('Found no model files, '
                                'please supply or mount a folder '
                                'containing h5 files with model weights.')
    notfound = []
    for model_file in predictor_files:
        if not os.path.exists(model_file):
            notfound.append(model_file)
    if notfound:
        raise FileNotFoundError('Some (or all) model files/folders were missing.\n'
                                'Please supply or mount a folder '
                                'containing the model files/folders with model weights.\n'
                                'Current problematic paths:\n\t' +
                                '\n\t'.join(notfound))

    for file in meta_data["files"]:
        print(args.model)
        path_file = os.path.join(args.model, file["name"])
        hashmd5 = md5(path_file)
        if file["md5"] != hashmd5:
            raise ValueError("Mismatch between expected file from the model descriptor and the actual model file")

    brainmask = args.braimask
    output_path = args.out_dir

    affine = None
    image_shape = None
    # Load brainmask if given (and get the affine & shape from it)
    if brainmask is not None:
        brainmask, aff = _load_image(brainmask)
        image_shape = brainmask.shape
        if affine is None:
            affine = aff

    # Load and/or build image from modalities
    image, aff = _load_image(args.img)
    if affine is None:
        affine = aff
    if image_shape is None:
        image_shape = image.shape
    else:
        if image.shape != image_shape:
            raise ValueError(
                f'Image and mask have different shape : {image_shape} vs {image.shape}'
            )
    if brainmask is not None:
        image *= brainmask
    images = []
    images.append(image)
    # Concat all modalities
    images = np.concatenate(images, axis=-1)
    # Add a dimension for a batch of one image
    images = np.reshape(images, (1,) + images.shape)

    chrono0 = time.time()
    # Load models & predict
    predictions = []
    for predictor_file in predictor_files:
        print(f"Loading predictor file: {predictor_file}")
        tf.keras.backend.clear_session()
        gc.collect()
        try:
            model = tf.keras.models.load_model(
                predictor_file,
                compile=False,
                custom_objects={"tf": tf})
        except Exception as err:
            print(f'\n\tWARNING : Exception loading model : {predictor_file}\n{err}')
            continue
        if hasattr(predictor_file, 'stem'):
            print('INFO : Predicting fold :', predictor_file.stem)
        prediction = model.predict(
            images,
            batch_size=1
        )
        # prediction = model(images, training=False)  # slightly (?) faster alternative
        if brainmask is not None:
            prediction *= brainmask
        predictions.append(prediction)

    # Average all predictions
    predictions = np.mean(predictions, axis=0)

    chrono1 = (time.time() - chrono0) / 60.
    if _VERBOSE:
        print(f'Inference time : {chrono1} sec.')

    # Threshold to remove near-zero voxels
    pred = predictions[0]
    pred[pred < 0.001] = 0

    # Save prediction
    nifti = nibabel.Nifti1Image(pred.astype('float32'), affine=affine)
    nibabel.save(nifti, output_path)

    if _VERBOSE:
        print(f'\nINFO : Done with predictions -> {output_path}\n')


if __name__ == "__main__":
    main()

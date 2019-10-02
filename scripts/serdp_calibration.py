#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
from serdpCalibrator.constants import Constants
from serdpCalibrator.serdp_calibrator import SerdpCalibrator, CheckerBoard
from stereoProcessing.intrinsic_extrinsic import Loader
from stereoProcessing.intrinsic_extrinsic import ExtrinsicIntrnsicLoaderSaver
import tifffile as tiff

import argparse
from os.path import dirname, abspath
import glob
import cv2
import numpy as np


def main(args):
    loader = Loader(base_path=args.base_path)
    loader.load_params_from_file(args.calibration_yaml)
    EI_loader = ExtrinsicIntrnsicLoaderSaver(loader)

    checkerboard = CheckerBoard()
    checkerboard.set_vals_from_yaml(args.checkerboard_yaml)

    calibrator = SerdpCalibrator(EI_loader, checkerboard=checkerboard)

    left_images = glob.glob(
        args.images_path + Constants.left_folder + "*" + args.encoding)
    right_images = glob.glob(
        args.images_path + Constants.right_folder + "*" + args.encoding)
    sonar_images = glob.glob(
        args.images_path + Constants.sonar_folder + "*" + ".tiff")

    images = zip(left_images, right_images, sonar_images)

    count = 0
    for left_name, right_name, sonar_name in images:
        left_img = cv2.imread(left_name)
        right_img = cv2.imread(right_name)
        sonar_img = tiff.imread(sonar_name)
        R, t, _ = calibrator.rigid_body_transform(left_img, right_img)
        N = calibrator.calculate_normal(R, t)
        x_pnts, z_pnts = calibrator.get_sonar_points(sonar_img)

        a = calibrator.construct_a_vector(N, x_pnts, z_pnts)
        print(type(a))
        b = np.array([np.linalg.norm(N)**2])
        if count == 0:
            A = a
            B = b
        else:
            A_ = np.zeros((A.shape[0] + a.shape[0], 9))
            A_[0:A.shape[0],:] = A
            A_[A.shape[0]:,:] = a
            A = A_
            B = np.concatenate(B, b)
        print(count, A.shape, B.shape)

        count += 1

    h = np.linalg.pinv(A)*B



if __name__ == '__main__':
    folder_path = dirname(dirname(abspath(__file__)))
    parser = argparse.ArgumentParser("Subscribe to images and save to folder")
    parser.add_argument(
        "--images_path", help="Folder with images",
        default=dirname(dirname(abspath(__file__))) + "/images/")
    parser.add_argument(
        "--encoding", help="Image encoding",
        default=".png")
    parser.add_argument(
        "--calibration_yaml",
        help="Path to calibration yaml specify path of calibration files",
        default=folder_path + "/cfg/calibrationConfig.yaml")
    parser.add_argument(
        "--base_path",
        help="Path to calibration yaml specify path of calibration files",
        default=folder_path + "/calibration/")
    parser.add_argument(
        "--checkerboard_yaml",
        help="Path to checkerboard description yaml file",
        default=folder_path + "/cfg/checkerboard.yaml")

    args = parser.parse_args()
    if args.images_path[-1] == "/":
        images_path = args.images_path
    else:
        images_path = args.images_path + "/"

    main(args)

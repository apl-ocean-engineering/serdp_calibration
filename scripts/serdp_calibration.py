#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
from serdpCalibrator.constants import Constants
from serdpCalibrator.serdp_calibrator import SerdpCalibrator, CheckerBoard
from stereoProcessing.intrinsic_extrinsic import Loader
from stereoProcessing.intrinsic_extrinsic import ExtrinsicIntrnsicLoaderSaver

import argparse
from os.path import dirname, abspath
import glob
import cv2


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
        args.images_path + Constants.sonar_folder + "*" + args.encoding)

    images = zip(left_images, right_images, sonar_images)

    for left_name, right_name, sonar_name in images:
        left_img = cv2.imread(left_name)
        right_img = cv2.imread(right_name)
        sonar_img = cv2.imread(sonar_name)
        RBT = calibrator.rigid_body_transform(left_img, right_img)
        #print("Point location", points4D)
        # cv2.imshow(Constants.fname1, left_img)
        # cv2.imshow(Constants.fname2, right_img)
        # cv2.imshow("frame3", sonar_img)
        cv2.waitKey(0)


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

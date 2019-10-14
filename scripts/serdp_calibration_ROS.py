#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
from serdpCalibrator.constants import Constants
from serdpCalibrator.serdp_calibrator import SerdpCalibrator, CheckerBoard, CalibrationPoint
from stereoProcessing.intrinsic_extrinsic import Loader
from stereoProcessing.intrinsic_extrinsic import ExtrinsicIntrnsicLoaderSaver
import tifffile as tiff
from sympy import Matrix
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
import copy

import argparse
from os.path import dirname, abspath
import glob
import cv2
import numpy as np

from itertools import permutations

class ROSCalibrator:
    def __init__(self):
        self.left_img = None
        self.right_img = None
        self.sonar_img = None
        self.bridge = CvBridge()
        rospy.Subscriber(
                        "/camera_left",Image, self.left_img_callback)
        rospy.Subscriber(
                        "/camera_right",Image, self.right_img_callback)
        rospy.Subscriber(
                        "/drawn_sonar",Image, self.sonar_callback)

    def left_img_callback(self, img):
        try:
          self.left_img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
          print(e)

    def right_img_callback(self, img):
        try:
          self.right_img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
          print(e)

    def sonar_callback(self, img):
        try:
          sonar_img = self.bridge.imgmsg_to_cv2(img, "16SC3")
          sonar_img.astype(np.uint16)
          self.sonar_img = sonar_img
        except CvBridgeError as e:
          print(e)

    def construct_A_B_matrices(self, calibration_points_lst, keep_list):
        A = np.zeros((len(keep_list), 9))
        B = np.zeros((len(keep_list), 1))
        err_lst = []

        row = 0

        for cp in calibration_points_lst:
            print(row, cp.id)
            if cp.id in keep_list:
                A[row, :] = cp.a
                B[row] = cp.b
                err_lst.append((cp.id, cp.error))

                row+=1
        return A, B, err_lst

    def main(self, args):
        loader = Loader(base_path=args.base_path)
        loader.load_params_from_file(args.calibration_yaml)
        EI_loader = ExtrinsicIntrnsicLoaderSaver(loader)

        checkerboard = CheckerBoard()
        checkerboard.set_vals_from_yaml(args.checkerboard_yaml)

        calibrator = SerdpCalibrator(EI_loader, checkerboard=checkerboard)

        rate = rospy.Rate(30)
        count = 0
        # B_lst = []
        # err_lst = []
        keep_list = []
        calibration_points_lst = []
        while not rospy.is_shutdown():
            left_img = self.left_img
            right_img = self.right_img
            sonar_img = self.sonar_img

            if left_img is not None and right_img is not None and sonar_img is not None:
                # try:
                    cv2.imshow(Constants.fname1, left_img)
                    cv2.imshow(Constants.fname2, right_img)
                    cv2.imshow("sonar_img", sonar_img)
                    k = cv2.waitKey(1)
                    if k == 13:  # enter
                        R, t, _, err = calibrator.rigid_body_transform(
                                                left_img, right_img)
                        N = calibrator.calculate_normal(R, t)
                        x_pnts, z_pnts = calibrator.get_sonar_points(sonar_img)
                        a = calibrator.construct_a_vector(N, x_pnts, z_pnts)
                        keep_list.append(count)

                        CP = CalibrationPoint()
                        CP.id = count
                        CP.error = err
                        CP.a = a
                        CP.b = np.linalg.norm(N)**2
                        calibration_points_lst.append(CP)

                        A, B, err_lst = self.construct_A_B_matrices(
                                calibration_points_lst, keep_list)
                        # print("A")
                        # print(A)
                        # print("B")
                        # print(B)
                        # print("condition number")
                        # print(calibrator.calculate_condition_number(A))

                        matrix_indicies = [i for i in keep_list]
                        combinations = calibrator.all_combinations(
                                                            matrix_indicies)
                        print("Count", count)
                        print("Keep list size", len(keep_list))

                        print("Rigid body transformation preprojection error")
                        print(sorted(err_lst, key = lambda x: x[1]))

                        calibrator.full_extrinsic_calculation(A, B)

                        save = input("correct? ")

                        if save == 1:
                            pass

                        elif save == 2:
                            while save == 2:
                                remove_nums = input("remove_nums: ")
                                for num in sorted(remove_nums, reverse = True):
                                    keep_list.remove(num)
                                A, B, err_lst = self.construct_A_B_matrices(
                                        calibration_points_lst, keep_list)
                                calibrator.full_extrinsic_calculation(
                                        A, B)
                                save = input(
                                "Correct? Press '2' to try new combination, \
                                or anything else to continue ")
                                if save == 2:
                                    for num in remove_nums:
                                        keep_list.append(num)
                                        keep_list = sorted(keep_list)
                        #
                        else:
                            keep_list.pop()

                        count += 1
            rate.sleep()



if __name__ == '__main__':
    rospy.init_node("serdp_calibrator")
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

    RC = ROSCalibrator()
    RC.main(args)

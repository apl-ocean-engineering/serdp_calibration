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

    def main(self, args):
        loader = Loader(base_path=args.base_path)
        loader.load_params_from_file(args.calibration_yaml)
        EI_loader = ExtrinsicIntrnsicLoaderSaver(loader)

        checkerboard = CheckerBoard()
        checkerboard.set_vals_from_yaml(args.checkerboard_yaml)

        calibrator = SerdpCalibrator(EI_loader, checkerboard=checkerboard)

        rate = rospy.Rate(30)
        count = 0
        B_lst = []
        err_lst = []
        keep_list = []
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
                        R, t, _, err = calibrator.rigid_body_transform(left_img, right_img)
                        err_lst.append([err, count])


                        N = calibrator.calculate_normal(R, t)

                        x_pnts, z_pnts = calibrator.get_sonar_points(sonar_img)

                        a = calibrator.construct_a_vector(N, x_pnts, z_pnts)
                        keep_list.append(count)

                        if count == 0:
                            A_ = a
                        else:
                            A_ = np.zeros((A.shape[0] + a.shape[0], 9))
                            A_[0:A.shape[0], :] = A
                            A_[A.shape[0]:, :] = a
                            A_ = A_[keep_list,:]


                        print("condition number")
                        print(calibrator.calculate_condition_number(A_))

                        matrix_indicies = [i for i in range(0, A_.shape[0])]
                        combinations = calibrator.all_combinations(matrix_indicies)
                        print("Count", count)

                        condition_number_lst = []
                        for combination in combinations:
                            if combination == []:
                                continue
                            if len(combination) >= A_.shape[0]:
                                continue
                            A_cond = calibrator.calculate_condition_number(
                                                 np.delete(A_, combination, 0))
                            score_inds = [len(combination), A_cond, combination]
                            condition_number_lst.append(copy.deepcopy(score_inds))

                        print("Condition numbers after removal")
                        downsampled_number_list = []
                        count_list = [0]*len(condition_number_lst)
                        for val in condition_number_lst:
                            ind = val[0]
                            count_list[ind -1] += 1
                            if count_list[ind -1] <= 5:
                                downsampled_number_list.append(val)

                        sorted_condition_list = sorted(downsampled_number_list, key=lambda tup: tup[1])
                        for val in sorted_condition_list:
                             print(count - val[0] + 1, val[0:])


                        print("Rigid body transformation preprojection error")
                        print(sorted(err_lst))

                        for i in range(0, len(x_pnts)):
                            B_lst.append(np.linalg.norm(N)**2)

                        B_ = []
                        #B_lst[keep_list]
                        for ind in keep_list:
                            B_.append(B_lst[ind])
                        calibrator.full_extrinsic_calculation(A_, B_)
                        save = input("correct? ")


                        if save == 1:
                            A = A_

                        elif save == 2:
                            A = A_
                            while save == 2:
                                # A_recalc = A_
                                # B_lst_recalc = B_lst
                                # err_lst_recalc = err_lst
                                remove_nums = input("remove_nums: ")
                                # A_recalc = np.delete(A, remove_nums, 0)
                                for num in sorted(remove_nums, reverse = True):
                                    keep_list.remove(num)
                                    # B_lst_recalc.pop(num)
                                    # err_lst_recalc.pop(num)
                                A_recalc = A_[keep_list, :]
                                B_lst_recalc = [B_[val] for val in keep_list]
                                calibrator.full_extrinsic_calculation(
                                        A_recalc, B_lst_recalc)
                                save = input(
                                "Correct? Press '2' to try new combination, or anything else to continue ")
                                if save != -2:
                                    for num in remove_nums:
                                        keep_list.append(num)
                                        keep_list = sorted(keep_list)
                        #
                        else:
                            keep_list.pop()
                            count -= 1

                        count += 1
                # except Exception as e:
                    # print(e)
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

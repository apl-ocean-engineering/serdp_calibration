#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu

Currently lots of testing, I'll clean it up...
"""
from serdpCalibrator.constants import Constants
from serdpCalibrator.serdp_calibrator import SerdpCalibrator, CheckerBoard
from stereoProcessing.intrinsic_extrinsic import Loader
from stereoProcessing.intrinsic_extrinsic import ExtrinsicIntrnsicLoaderSaver
import tifffile as tiff
from sympy import Matrix

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

    left_images = sorted(glob.glob(
        args.images_path + Constants.left_folder + "*" + args.encoding))
    right_images = sorted(glob.glob(
        args.images_path + Constants.right_folder + "*" + args.encoding))
    sonar_images = sorted(glob.glob(
        args.images_path + Constants.sonar_folder + "*" + ".tiff"))

    images = zip(left_images, right_images, sonar_images)

    count = 0
    B_lst = []
    for left_name, right_name, sonar_name in images:
        # if count > 9:
            # break
        ###
        # left_name = left_images[0]
        # right_name = right_images[0]
        # sonar_name = sonar_images[0]
        ###
        left_img = cv2.imread(left_name)
        right_img = cv2.imread(right_name)
        sonar_img = tiff.imread(sonar_name)
        R, t, _, err = calibrator.rigid_body_transform(left_img, right_img)
        # print("R")
        # print(R)
        # print("t")
        # print(t)
        # t_hat = np.matmul(R,t)
        # print("stereo estimated point")
        # print(t_hat)
        # print("error")
        # print(err)

        N = calibrator.calculate_normal(R, t)
        # print("N")
        # print(N)
        # print("Norm N", np.linalg.norm(N)**2)
        x_pnts, z_pnts = calibrator.get_sonar_points(sonar_img)
        # print("sonar estimated point")
        # print(x_pnts[0],z_pnts[0])

        a = calibrator.construct_a_vector(N, x_pnts, z_pnts)
        # print("a", a)
        # print("|N|**2", np.linalg.norm(N)**2)
        # b = np.array([np.linalg.norm(N)**2])
        if count == 0:
            A_ = a
            # B = b
        else:
            A_ = np.zeros((A.shape[0] + a.shape[0], 9))
            A_[0:A.shape[0], :] = A
            A_[A.shape[0]:, :] = a
            # A = A_
        count += 1

        # print("A shape", A_.shape)
        # if A_.shape == (9,9):
            # print("INV", np.linalg.inv(A_))
        # print("A", A_)
        # print("rref")
        # M = Matrix(A_.reshape((A_.shape[0],9)))
        # print("M")
        # print(M)
        # print("M size")
        # print(M.shape)
        # print("rref vals")
        # print(M.rref())
        print("Condition number")
        print(np.linalg.cond(A_))
        # for i in range(A_.shape[0]):
        #     if i == 0:
        #         M = Matrix(A_[i,:].reshape(1,9))
        #
        #     else:
        #         M.row_insert(0, A[i,:].reshape(1,9))
        #     print(M)
        # print("det", np.linalg.det(np.matmul(A_, A_.T)))
        # print("count", count)
        # print("A pinv", np.linalg.pinv(A_))
        # B_ = np.ones((A.shape[0], 1))
        # B = np.multiply(B_, [np.linalg.norm(N)**2])
        for i in range(0, len(x_pnts)):
            B_lst.append(np.linalg.norm(N)**2)

        # print("B_lst", np.array(B_lst).reshape((A_.shape[0],)))
        h = np.matmul(np.linalg.pinv(A_), np.array(B_lst).reshape((A_.shape[0],)))
        print("h")
        print(h)
        for i in range(A_.shape[0]):
            pass
            # print("N**2", B_lst[i])
            # print("h*a", np.matmul(A_[i,:], h.transpose()))
        # print(np.linalg.det(np.matmul(A, A.transpose())))
        # print(np.linalg.inv(np.matmul(A, A.transpose())))

        H = h.reshape((3, 3))

        R[:, 0] = H[:, 0]
        R[:, 1] = np.cross(-H[:, 0], H[:, 1])
        R[:, 2] = H[:, 1]
        R = R.transpose()
        t = np.matmul(-R, H[:, 2])
        # print("R,t", R,t)
        # print("A A.T")
        # print(np.matmul(A_, A_.T))
        # print("inv(A A.T)")
        # print(np.linalg.inv(np.matmul(A_, A_.T)))
        # print("pinv(A)")
        # print(np.linalg.pinv(A_))
        # print("A*AT * inv(A*AT) == I check")
        # inv = (np.linalg.inv(np.matmul(A_, A_.T)))
        # print(np.matmul(inv, np.matmul(A_, A_.T)))
        # print("find r")
        # print(calibrator._find_R(H))
        # print()
        # print("SVD adjustment")
        u, s, vh = np.linalg.svd(R)
        R = np.matmul(u, vh)
        # print(R)

        save = input("correct? ")
        if save == 1:
            A = A_
        elif save == 2:
            A = A_
            remove_nums = input("remove_nums: ")
            for num in remove_nums:
                np.delete(A, num, 0)
                del B_lst[num]
        else:
            B_lst.pop()
            count -=1
            # print("retry")
            h = np.matmul(np.linalg.pinv(A), np.array(B_lst).reshape((A.shape[0],)))
            # print("h", h)


    B_ = np.ones((A.shape[0], 1))
    B = np.multiply(B_, [np.linalg.norm(N)**2])
    print(count, A.shape, B.shape)


    h = np.matmul(np.linalg.pinv(A), B)
    print(h)

    H = h.reshape((3, 3))
    R = np.zeros((3, 3))
    R[:, 0] = H[:, 0]
    R[:, 1] = np.cross(-H[:, 0], H[:, 1])
    R[:, 2] = H[:, 1]
    R = R.transpose()
    t = np.matmul(-R, H[:, 2])
    print(R,t)



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

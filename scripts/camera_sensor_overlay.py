#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
from serdpCalibrator.constants import Constants
from stereoProcessing.intrinsic_extrinsic import Loader
from stereoProcessing.intrinsic_extrinsic import ExtrinsicIntrnsicLoaderSaver
import glob
import cv2
import tifffile as tiff
import numpy as np
import yaml

from cv_bridge import CvBridge, CvBridgeError

import image_geometry

import rospy
from sensor_msgs.msg import PointCloud, Image, CameraInfo
from geometry_msgs.msg import Point32
from sensor_msgs.msg import ChannelFloat32
import std_msgs.msg

import argparse
from os.path import dirname, abspath

SONAR_VERT_FOV = 4.5  # degrees
SCALE_FACTOR = 250.0
Z_MIN = 0.1
Z_MAX = 0.9
ITENSITY_MIN = 10000

def _polar_to_cartesian(bearing, range):
    x = range * -np.cos(np.radians(bearing))
    z = range * np.sin(np.radians(bearing))

    return x, z


def sonar_mouse_click(event, x, y, flags, param):
    """
    Callback function for mouse click event on image1 frame

    Places clicked points into x1_ and y1_points lists
    """
    bearing = param[y, x, 0]/SCALE_FACTOR
    range = param[y, x, 1]/SCALE_FACTOR
    if event == cv2.EVENT_LBUTTONDOWN:
        print(param[y, x])
        x, z = _polar_to_cartesian(bearing, range)
        print(x, z)
        print(bearing, range)

def construct_info_msg(EI_loader, im_size, name="left"):
    info = CameraInfo()
    info.height = im_size[0]
    info.width = im_size[1]
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    info.distortion_model = "plumb_bob"
    if name == "left":
        info.D = EI_loader.paramaters.d1.reshape(1,5).tolist()[0]
        info.K = EI_loader.paramaters.K1.reshape(1,9).tolist()[0]
        info.R = EI_loader.paramaters.R1.reshape(1,9).tolist()[0]
        info.P = EI_loader.paramaters.P1.reshape(1,12).tolist()[0]
        header.frame_id = 'camera_left'
        info.header = header

    if name == "right":
        info.D = EI_loader.paramaters.d2.reshape(1,5).tolist()[0]
        info.K = EI_loader.paramaters.K2.reshape(1,9).tolist()[0]
        info.R = EI_loader.paramaters.R2.reshape(1,9).tolist()[0]
        info.P = EI_loader.paramaters.P2.reshape(1,12).tolist()[0]
        header.frame_id = 'camera_right'
        info.header = header

    return info, header


def y_range(ranges, fov):
    y1 = np.float32(ranges)*np.radians(fov)

    return y1, -y1

def polar_to_cartesian(bearings, ranges):
    x = np.multiply(ranges/SCALE_FACTOR, -np.cos(np.radians(bearings/SCALE_FACTOR)))
    z = np.multiply(ranges/SCALE_FACTOR, np.sin(np.radians(bearings/SCALE_FACTOR)))

    return x, z

def constract3D_sonar_array(sonar_img):
    y_nums = 10
    intensity_lst = []
    count = 0
    bearings = sonar_img[:,:,0]
    ranges = sonar_img[:,:,1]
    intensity = sonar_img[:,:,2]
    intensity_lst = []

    x,z = polar_to_cartesian(bearings, ranges)

    y = np.zeros(x.shape)

    sonar_info = np.array([x,y,z,intensity]).T
    sonar_img_3D = np.zeros((sonar_info.shape[0], sonar_info.shape[1], 3))

    for i in range(sonar_info.shape[0]):
        for j in range(sonar_info.shape[1]):
            val = sonar_info[i][j]
            sonar_img_3D[i][j] = val[0:3]
            intensity_lst.append(float(val[3]))
    sonar_img_3D = sonar_img_3D.reshape(
                         (sonar_img_3D.shape[0]*sonar_img_3D.shape[1], 3))

    return sonar_img_3D, intensity_lst


def proj_sonar_image(EI_loader, sonar_img_3D, intensity_lst, im_size, img, R = None, t = None):
    axis = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                      [0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1] ])

    if R is None or t is None:
        r = np.float64(np.eye(3))
        t = np.float64(np.zeros((1,3)))
    else:
        r = np.float64(cv2.Rodrigues(R)[0])
        t = np.float64(t)

    sonar_img_3D = np.float64(sonar_img_3D)

    i = 0
    additional_points_intensity = []
    for count, point in enumerate(sonar_img_3D):
        x = point[0]
        y = point[1]
        z = point[2]
        intensity = intensity_lst[count]
        if z < Z_MAX and intensity > ITENSITY_MIN:
            current_pose = np.array([x,y,z]).reshape(1,3)
            y1, y2 = y_range(z, SONAR_VERT_FOV)
            translation = np.array([0, y1, 0])
            additional_points_intensity.append(intensity)
            R= np.eye(3)
            motion = np.array([np.matmul(R, translation)]).reshape(1,3)
            if i == 0:
                upper_point = current_pose - motion
                lower_point = current_pose + motion
            else:
                upper_point = np.append(upper_point, np.add(current_pose, -motion), axis = 0)
                lower_point = np.append(lower_point, np.add(current_pose, motion), axis = 0)
            i+=1

    imgpoints, jac = cv2.projectPoints(
        sonar_img_3D, r, t,  np.float64(EI_loader.paramaters.K1),
        np.float64(EI_loader.paramaters.d1))

    upper_imgpoints, jac = cv2.projectPoints(
        np.float64(upper_point), r, t,  np.float64(EI_loader.paramaters.K1),
        np.float64(EI_loader.paramaters.d1))

    lower_imgpoints, jac = cv2.projectPoints(
        np.float64(lower_point), r, t,  np.float64(EI_loader.paramaters.K1),
        np.float64(EI_loader.paramaters.d1))

    projected_sonar_img = np.zeros((im_size[0], im_size[1], 3))
    for count, point in enumerate(imgpoints):
        intensity = intensity_lst[count]
        x = point[0][0]
        y = point[0][1]
        z = sonar_img_3D[count][2]

        if x > 0 and x < im_size[1] and y > 0 and y < im_size[0] and z < Z_MAX and intensity > ITENSITY_MIN:
            projected_sonar_img[int(y), int(x), 2] = intensity/SCALE_FACTOR

    for count, point in enumerate(upper_imgpoints):
        intensity = additional_points_intensity[count]
        x = point[0][0]
        y = point[0][1]
        if x > 0 and x < im_size[1] and y > 0 and y < im_size[0]:
            projected_sonar_img[int(y), int(x), 0] = 255


    for count, point in enumerate(lower_imgpoints):
        intensity = additional_points_intensity[count]
        x = point[0][0]
        y = point[0][1]
        z = sonar_img_3D[count][2]
        if x > 0 and x < im_size[1] and y > 0 and y < im_size[0]:
            projected_sonar_img[int(y), int(x), 1] = intensity/SCALE_FACTOR

    projected_sonar_img = np.uint8(projected_sonar_img)
    kernel = np.ones((5,5), np.uint8)
    projected_sonar_img = cv2.dilate(projected_sonar_img, kernel)
    display_img = cv2.addWeighted(projected_sonar_img,0.5,img,0.5,0)


    return display_img, projected_sonar_img

def main(args):

    loader = Loader(base_path=args.base_path)
    loader.load_params_from_file(args.calibration_yaml)
    EI_loader = ExtrinsicIntrnsicLoaderSaver(loader)

    left_images = sorted(glob.glob(
        args.images_path + Constants.left_folder + "*" + args.encoding))
    right_images = sorted(glob.glob(
        args.images_path + Constants.right_folder + "*" + args.encoding))
    sonar_images = sorted(glob.glob(
        args.images_path + Constants.sonar_folder + "*" + ".tiff"))


    left_img = cv2.imread(left_images[0])
    right_img = cv2.imread(right_images[0])
    sonar_img = tiff.imread(sonar_images[0])

    im_size = left_img.shape

    left_info = construct_info_msg(EI_loader, im_size, name="left")
    right_info = construct_info_msg(EI_loader, im_size, name="right")

    sonar_img_3D, intensity = constract3D_sonar_array(sonar_img)

    with open(args.sonar_stereo_extrinsics, 'r') as stream:
        calibration_loader = yaml.safe_load(stream)

    R_loader = calibration_loader["rotation_matrix"]
    R = R_loader["data"]
    R = np.array([R]).reshape((R_loader["rows"], R_loader["cols"]))

    t_loader = calibration_loader["translation"]
    t = t_loader["data"]
    t = np.array([t]).reshape((t_loader["rows"], t_loader["cols"]))

    # print(R)
    # print(t)
    # print(cv2.Rodrigues(R)[0])

    t_shift = np.array([[0.0,0.0,0.0]]).transpose()#np.array([[-0.05,-0.15,-0.0]]).transpose()

    # print(np.add(t,t_shift))

    R = R.transpose()
    t = np.matmul(-R,np.add(t,t_shift))


    rot_display_img, rot_projected_sonar_img = proj_sonar_image(
        EI_loader, sonar_img_3D, intensity, im_size, left_img.copy(), R, t)

    # norm_display_img, norm_projected_sonar_img = proj_sonar_image(
    #     EI_loader, sonar_img_3D, intensity, im_size, left_img.copy(), np.float64(np.eye(3)), t)
    #
    # all_norm_display_img, all_norm_projected_sonar_img = proj_sonar_image(
    #     EI_loader, sonar_img_3D, intensity, im_size, left_img.copy())

    # np.save("img.npy", sonar_img_3D)
    # np.save("intensity.npy",np.array(intensity))

    cv2.imshow("rot_display_img", rot_display_img)
    cv2.imshow("rot_projected_sonar_img", rot_projected_sonar_img)
    # cv2.imshow("norm_display_img", norm_display_img)
    # cv2.imshow("norm_projected_sonar_img", norm_projected_sonar_img)
    # cv2.imshow("all_norm_display_img", all_norm_display_img)
    # cv2.imshow("all_norm_projected_sonar_img", all_norm_projected_sonar_img)


    cv2.imshow("sonar image", sonar_img)
    cv2.setMouseCallback("sonar image", sonar_mouse_click, param = sonar_img)


    cv2.waitKey(0)

if __name__ == '__main__':
    rospy.init_node("sonar_pc")

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
        "--sonar_stereo_extrinsics",
        help="Path to sonaral cmaera extrinsics",
        default=folder_path + "/cfg/stereo_sonar_extrinsics2.yaml")

    args = parser.parse_args()
    if args.images_path[-1] == "/":
        images_path = args.images_path
    else:
        images_path = args.images_path + "/"

    main(args)

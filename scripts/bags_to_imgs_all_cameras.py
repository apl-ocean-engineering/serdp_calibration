#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import argparse
from os.path import dirname, abspath
from serdpCalibrator.constants import Constants
import numpy as np
import tifffile as tiff

SCALE_FACTOR = 250


class ImageSaver:
    def __init__(self, save_path):
        self.bridge = CvBridge()
        self.left_sub = rospy.Subscriber(
            "camera_left", Image, self.left_callback)
        self.right_sub = rospy.Subscriber(
            "camera_right", Image, self.right_callback)

        self.left_img = None
        self.right_img = None

        self.save_path = save_path

    def left_callback(self, data):
        try:
            self.left_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def right_callback(self, data):
        try:
            self.right_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def main(self):
        rate = rospy.Rate(30)
        count = 0
        while not rospy.is_shutdown():
            if self.left_img is not None and self.right_img is not None:
                if self.left_img.shape[0] > 0 and self.right_img.shape[0] > 0:
                    cv2.imshow(Constants.fname1, self.left_img)
                    cv2.imshow(Constants.fname2, self.right_img)

                    cv2.waitKey(3)
                    left_name = self.save_path + \
                        Constants.left_folder + "left" + \
                        str(count) + ".png"
                    right_name = self.save_path + \
                        Constants.right_folder + "right" + \
                        str(count) + ".png"



                    cv2.imwrite(left_name, self.left_img)
                    cv2.imwrite(right_name, self.right_img)
                    count += 1

            rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Subscribe to images and save to folder")
    parser.add_argument(
        "--save_path", help="Folder to save images",
        default=dirname(dirname(abspath(__file__))) + "/images")
    parser.add_argument(
        "--base_path", help="Base folder to calibration values",
        default=dirname(dirname(abspath(__file__))) + "/calibration")

    args = parser.parse_args()

    rospy.init_node("bag_to_png")
    if args.save_path[-1] == "/":
        save_path = args.save_path
    else:
        save_path = args.save_path + "/"

    IS = ImageSaver(save_path)
    IS.main()

#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
import copy
from stereoProcessing.point_identification3 import PointIdentification3D
import yaml
import numpy as np
import cv2
import sys

sonar_img_name = "sonar_img"
range_conversion = 1.0  # assuming m...

SCALE_FACTOR = 250.0


class CheckerBoard:
    def __init__(self):
        self.lower_x = 0.0
        self.lower_y = 0.0
        self.width = 0.0
        self.height = 0.0
        self.z = 0.0

    def set_vals_from_yaml(self, yaml_file):
        with open(yaml_file, 'r') as stream:
            calibration_loader = yaml.safe_load(stream)

        self._set_vals(
            float(calibration_loader['lower_x']),
            float(calibration_loader['lower_y']),
            float(calibration_loader['width']),
            float(calibration_loader['height']),
            float(calibration_loader['z']))

    def set_vals_mannually(self, lower_left=(0, 0), width=0, height=0, z=0):
        self._set_vals(lower_left[0], lower_left[1], width, height, z)

    def get_poses(self):
        pnts = np.array([self._lower_left, self._upper_left,
               self._upper_right, self._lower_right]).transpose()

        return pnts

    @property
    def _lower_left(self):
        return self.lower_left

    @property
    def _upper_left(self):

        return np.add(self.lower_left, np.array([0, self.height, 0]))

    @property
    def _upper_right(self):
        return np.add(self.lower_left, np.array([self.width, self.height, 0]))

    @property
    def _lower_right(self):

        return np.add(self.lower_left, np.array([self.width, 0, 0]))

    def _set_vals(self, lower_x, lower_y, width, height, z):
        self.lower_left = np.array([lower_x, lower_y, z])
        self.width = width
        self.height = height


class SerdpCalibrator:
    def __init__(self, EI_loader, checkerboard=CheckerBoard()):
        self.EI_loader = EI_loader
        self.checkerboard = checkerboard
        self.sonar_img = None
        self.sonar_x_points = []
        self.sonar_z_points = []


    def rigid_body_transform(self, left_img, right_img):
        """
        Calculates rigid body transformation of checkboard. Points are
        identified through stereo triangulation, and then a rigid body
        transform is calculated between the current checkerboard pose
        and the original checkerboard pose with default left corner at (0,0,0)

        Return:
            R (np array): Rotation matrix
            t (np array): Translation matrix
            lower_left_3D_position (np array): Position of lower left point
        """
        self.PI = PointIdentification3D(self.EI_loader)
        points4D = self.get_img_points(left_img, right_img)
        lower_left_3D_position = points4D[:3, 1]
        print(points4D)

        R, t = self._calculate_rigid_body(
            self.checkerboard.get_poses(), points4D[:3, :])

        return R, t, lower_left_3D_position

    def calculate_normal(self, R, t):
        R_3 = R[:, 2]  # third column of R
        N = R_3*np.matmul(R_3.transpose(), -t)

        return N

    def get_img_points(self, left_img, right_img):
        points4D = self.PI.get_points(
            copy.copy(left_img), copy.copy(right_img))
        points4D /= points4D[3]

        print(points4D)

        return points4D

    def get_sonar_points(self, sonar_img):

        cv2.namedWindow(sonar_img_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(sonar_img_name, self._sonar_mouse_click)
        self.sonar_x_points = []
        self.sonar_z_points = []
        self.sonar_img = sonar_img
        cv2.imshow(sonar_img_name, self.sonar_img)
        k = 0
        while k != 13: #enter
            k = cv2.waitKey(0)
            if k == 113: #q
                cv2.destroyAllWindows()
                sys.exit()
        cv2.destroyAllWindows()

        return self.sonar_x_points, self.sonar_z_points

    def construct_a_vector(self, N, x_pnts, z_pnts):
        a = np.zeros((len(x_pnts), 9))
        n1 = N[0]
        n2 = N[1]
        n3 = N[2]
        for i in range(0, len(x_pnts)):
            x = x_pnts[i]
            z = z_pnts[i]
            a_i = np.array(
                [n1*x, n1*z, n1, n2*x, n2*z, n2, n3*x, n3*z, n3])
            a[i] = a_i

        return a

    def _calculate_rigid_body(self, pnts1, pnts2):
        c1 = self._find_centroid(pnts1)
        c2 = self._find_centroid(pnts2)
        H = self._find_covariance_matrix(pnts1, c1, pnts2, c2)
        R = self._find_R(H)
        t = np.add(np.matmul(-R, c1), c2)

        return R, t

    def _find_covariance_matrix(self, pnts1, c1, pnts2, c2):
        H = np.zeros((3, 3))
        for i in range(0, pnts1.shape[1]):
            pa = np.subtract(pnts1[:, i], c1).reshape(3, 1)
            pb = np.subtract(pnts2[:, i], c2).reshape(3, 1)
            H = np.add(H, np.matmul(pa, pb.transpose()))

        return H

    def _find_R(self, H):
        u, s, vh = np.linalg.svd(H)
        R = np.matmul(u, vh)
        if np.linalg.det(R) < 0:
            R *= -1

        return R

    def _find_centroid(self, pnts):
        pnt = np.array([0, 0, 0])
        for i in range(0, pnts.shape[1]):
            pnt = np.add(pnt, pnts[:, i])
        pnt /= i

        return pnt

    def _polar_to_cartesian(self, bearing, range):
        x = range * -np.cos(np.radians(bearing))
        z = range * np.sin(np.radians(bearing))

        return x, z

    def _sonar_mouse_click(self, event, x, y, flags, param):
        """
        Callback function for mouse click event on image1 frame

        Places clicked points into x1_ and y1_points lists
        """
        if event == cv2.EVENT_LBUTTONDOWN and self.sonar_img is not None:
            bearing = self.sonar_img[y, x, 0]/SCALE_FACTOR
            range = self.sonar_img[y, x, 1]/SCALE_FACTOR
            cv2.circle(self.sonar_img, (x, y), 7, (255, 0, 0), -1)
            cv2.imshow(sonar_img_name, self.sonar_img)

            x, z = self._polar_to_cartesian(bearing, range)
            print(bearing, range)
            self.sonar_x_points.append(x)
            self.sonar_z_points.append(z)
            # Draw circle where clicked

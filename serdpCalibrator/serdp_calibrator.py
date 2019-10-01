#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
import copy
from stereoProcessing.point_identification3 import PointIdentification3D
import yaml
import numpy as np


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
        self.PI = PointIdentification3D(EI_loader)
        self.checkerboard = checkerboard

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
        points4D = self.get_img_points(left_img, right_img)
        lower_left_3D_position = points4D[:3, 1]

        R, t = self._calculate_rigid_body(
            self.checkerboard.get_poses(), points4D[:3, :])

        return R, t, lower_left_3D_position

    def get_img_points(self, left_img, right_img):
        points4D = self.PI.get_points(
            copy.copy(left_img), copy.copy(right_img))
        points4D /= points4D[3]

        return points4D

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

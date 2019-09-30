#!/usr/bin/env python2.7

"""
@author: Mitchell Scott
@contact: miscott@uw.edu
"""
from constants import Constants
import argparse
from os.path import dirname, abspath
import glob


def main(path, encoding):
    left_images = glob.glob(path + Constants.left_folder + "/*" + encoding)
    right_images = glob.glob(path + Constants.left_folder + "/*" + encoding)
    sonar_images = glob.glob(path + Constants.left_folder + "/*" + encoding)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Subscribe to images and save to folder")
    parser.add_argument(
        "--images_path", help="Folder with images",
        default=dirname(dirname(abspath(__file__))) + "/images")
    parser.add_argument(
        "--encoding", help="Image encoding",
        default=dirname(dirname(abspath(__file__))) + ".png")

    args = parser.parse_args()
    if args.images_path[-1] == "/":
        images_path = args.save_path
    else:
        images_path = args.images_path + "/"

    main(images_path)

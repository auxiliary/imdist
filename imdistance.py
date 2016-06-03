#!/usr/bin/env python

import numpy as np
import sys
import cv2
import os
import shutil
import glob
import math
from matcher import *
from matplotlib import pyplot as plt

def imdistance(filename1, filename2, show = None):
    if show == None:
        show = False

    img1 = cv2.imread(filename1, 0) # Target image
    img2 = cv2.imread(filename2, 0)

    orb = cv2.ORB()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    print "Number of matches: {}, len of des1 {}, len of des2 {}".format(len(matches), len(des1), len(des2))

    sum = 0

    def get_euclidean_dist_to_all(_id, keypoints):
        total_distance = 0
        for point in keypoints:
            total_distance += math.sqrt((keypoints[_id].pt[0] - point.pt[0]) ** 2 + (keypoints[_id].pt[1] - point.pt[1]) ** 2)

        return total_distance

    for match in matches:
        sum += abs(get_euclidean_dist_to_all(match.queryIdx, kp1) - get_euclidean_dist_to_all(match.trainIdx, kp2))

    if show == True:
        print sum
        img3 = drawMatches(img1, kp1, img2, kp2, matches[:])

    return sum

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage:\t./imdistance <filename 1> <filename 2> \n\t./imdistance <filename 1> <directory of images>'
        exit(-1)

    filename1 = sys.argv[1]
    filename2 = sys.argv[2]

    if os.path.isdir(filename2):
        path = filename2
        sorted_path = path + '/SORTED/'
        if not os.path.exists(sorted_path):
            os.mkdir(sorted_path)

        distances = []
        for filename in glob.glob(path + '/images*'):
            try:
                distance = imdistance(filename1, filename)
                distances.append((filename, distance))
                shutil.copyfile(filename, sorted_path + '/' + str(distance) + '.jpg')
                print 'Succeeded for {} with distance {}'.format(filename, distance)
            except:
                print 'Failed for {}'.format(filename)
                #exit()
                pass
            
    else:
        imdistance(filename1, filename2, show=True)


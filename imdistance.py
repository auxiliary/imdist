#!/usr/bin/env python

import numpy as np
import imagehash
import math
import sys
import cv2
import os
import shutil
import glob
import math
from matcher import *
from matplotlib import pyplot as plt


# Let's say des1 (img1) is the target image that's not changing
# So let's see how much of its features are in img1 and to what degree
def matched(_id, matches):
    for match in matches:
        if match.queryIdx == _id:
            return match
    return False

def avg_hamming_metric(bf, candid, other):
    distance_matrix = np.zeros(( len(candid['descriptors']), len(other['descriptors']) ))
    for i, descriptor in enumerate(candid['descriptors']):
        for j, other_descriptor in enumerate(other['descriptors']): 
            temp = bf.match( np.array([ descriptor ]), np.array([ other_descriptor ]) )
            distance_matrix[i, j] = temp[0].distance

    distance_matrix_copy = distance_matrix.copy()
    # best_matches for the one that has the minimum number of descriptors (for the candid)
    best_matches = []
    FILTER_FILL = 900000 # A number that replaces the row and column of a minimum
    for i in range(len(candid['descriptors'])):
        index = np.argmin(distance_matrix)
        index_2d = np.unravel_index(index, distance_matrix.shape)
        val = distance_matrix[index_2d]
        distance_matrix[index_2d[0], :] = FILTER_FILL
        distance_matrix[:, index_2d[1]] = FILTER_FILL
        best_matches.append((index_2d[0], index_2d[1], val))

    return np.mean(best_matches, axis=1)[2], best_matches, distance_matrix_copy


def euclidean_distance(candid, other, candid_id, other_id, matches):
    sum_candid = 0
    sum_other = 0

    for i, _, _ in matches:
        distance = math.sqrt((candid['keypoints'][i].pt[0] - candid['keypoints'][candid_id].pt[0]) ** 2 + (candid['keypoints'][i].pt[1] - candid['keypoints'][candid_id].pt[1]) ** 2)
        sum_candid += distance

    for _, i, _ in matches:
        distance = math.sqrt((other['keypoints'][i].pt[0] - other['keypoints'][other_id].pt[0]) ** 2 + (other['keypoints'][i].pt[1] - other['keypoints'][other_id].pt[1]) ** 2)
        sum_other += distance

    return abs(sum_candid - sum_other)


# Adjusts the hamming metric based on euclidean distances
def adjust_hamming_metric(candid, other, matches):
    adjusted_matches = []
    for i, j, val in matches:
        distance = euclidean_distance(candid, other, i, j, matches)
        adjusted_matches.append((i, j, distance * val))

    return np.mean(adjusted_matches, axis=1)[2]


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

    # Let's get the descriptor ratio
    des_ratio = abs(len(des1) - len(des2)) / float(len(des1))
    print 'Descriptor ratio metric: {}'.format(des_ratio)

    # Find minimum number of descriptors between the two images
    if len(des1) < len(des2):
        candid = {'keypoints': kp1, 'descriptors': des1}
        other =  {'keypoints': kp2, 'descriptors': des2}
    else:
        candid = {'keypoints': kp2, 'descriptors': des2}
        other =  {'keypoints': kp1, 'descriptors': des1}


    avg_hamming, best_matches, distance_matrix = avg_hamming_metric(bf, candid, other)
    avg_hamming = adjust_hamming_metric(candid, other, best_matches)

    #print 'Average Hamming metric: {}'.format(avg_hamming)

    if (len(des2) < 30):
        avg_hamming = 900000

    if show == True:
        print avg_hamming
        img3 = drawMatches(img1, kp1, img2, kp2, matches[:])

    return avg_hamming

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
        for filename in glob.glob(path + '/*.png'):
            try:
                distance = imdistance(filename1, filename)
                distances.append((filename, distance))
                shutil.copyfile(filename, sorted_path + '/' + str(distance) + '.png')
                print 'Succeeded for {} with distance {}'.format(filename, distance)
            except:
                print 'Failed for {}'.format(filename)
                #exit()
                pass
            
    else:
        imdistance(filename1, filename2, show=True)


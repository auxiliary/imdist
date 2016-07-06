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

# ROB
import random


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

# ROB
def avg_hamming_metric2(candid, other):
    minD = -1
    maxD = -1
    distance_matrix = np.zeros(( len(candid['descriptors']), len(other['descriptors']) ))
    for i, d1 in enumerate(candid['descriptors']):
        for j, d2 in enumerate(other['descriptors']): 
            theD = cv2.norm( d1, d2, cv2.NORM_HAMMING);
            distance_matrix[i, j] = theD

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

# ROB
def per_image_stats(inp):
    minH = -1
    maxH = -1
    minP = -1
    maxP = -1
    for i, (d1, k1) in enumerate(zip(inp['descriptors'], inp['keypoints'])):
        for j, (d2, k2) in enumerate(zip(inp['descriptors'], inp['keypoints'])):
            if(i == j):
                continue
            distH = cv2.norm( d1, d2, cv2.NORM_HAMMING);
            if(distH < minH or minH == -1):
                minH = distH
            if(distH > maxH or maxH == -1):
                maxH = distH

            distP = math.sqrt((k1.pt[0] - k2.pt[0]) ** 2 + (k1.pt[1] - k2.pt[1]) ** 2)
            if(distP < minP or minP == -1):
                minP = distP
            if(distP > maxH or maxP == -1):
                maxP = distP
    met=[]
    for i, (d1, k1) in enumerate(zip(inp['descriptors'], inp['keypoints'])):
        met.append(0)
        for j, (d2, k2) in enumerate(zip(inp['descriptors'], inp['keypoints'])):
            if(i == j):
                continue
            distH = (cv2.norm( d1, d2, cv2.NORM_HAMMING)-minH)/(maxH-minH);
            distP = (math.sqrt((k1.pt[0] - k2.pt[0]) ** 2 + (k1.pt[1] - k2.pt[1]) ** 2)-minP)/(maxP-minP)
            met[i] += math.sqrt(distH ** 2 + distP ** 2)

    return met

def new_metric(matches, m1, m2):
    adjusted_matches = 0

    for i, j, val in matches:
        adjusted_matches += abs(m1[i]-m2[j])
    avgM = adjusted_matches/len(matches)

    return avgM
# END ROB

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

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # ROB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #matches = bf.match(des1, des2)

    #print "Number of matches: {}, len of des1 {}, len of des2 {}".format(len(matches), len(des1), len(des2))

    # Let's get the descriptor ratio
    #des_ratio = abs(len(des1) - len(des2)) / float(len(des1))
    #print 'Descriptor ratio metric: {}'.format(des_ratio)

    # Find minimum number of descriptors between the two images
    #if len(des1) < len(des2):
    #    candid = {'keypoints': kp1, 'descriptors': des1}
    #    other =  {'keypoints': kp2, 'descriptors': des2}
    #else:
    #    candid = {'keypoints': kp2, 'descriptors': des2}
    #    other =  {'keypoints': kp1, 'descriptors': des1}
    if len(des1) < len(des2):
        candid = {'keypoints': kp1, 'descriptors': des1}
        #dS, kS = zip(*random.sample(list(zip(des2, kp2)), len(des1)))
        dS=[]
        kS=[]
        sample = sorted(random.sample(range(len(des2)), len(des1)))
        for s in sample:
            dS.append(des2[s])
            kS.append(kp2[s])
        other =  {'keypoints': kS, 'descriptors': dS}
        numSamples = len(des1)
    else:
        dS=[]
        kS=[]
        sample = sorted(random.sample(range(len(des1)), len(des2)))
        for s in sample:
            dS.append(des1[s])
            kS.append(kp1[s])
        #dS, kS = zip(*random.sample(zip(des1, kp1), len(des2)))
        candid =  {'keypoints': kS, 'descriptors': dS}
        other = {'keypoints': kp2, 'descriptors': des2}
        numSamples = len(des2)

    #avg_hamming, best_matches, distance_matrix = avg_hamming_metric(bf, candid, other)
    #avg_hamming = adjust_hamming_metric(candid, other, best_matches)
    avg_hamming, best_matches, distance_matrix = avg_hamming_metric2(candid, other)
    met1 = per_image_stats(candid)
    met2 = per_image_stats(other)

    avg_hamming = new_metric(best_matches, met1, met2)

    #print 'Average Hamming metric: {}'.format(avg_hamming)
    print 'Fancy new metric: {}'.format(avg_hamming)

    #avg_hamming *= (len(des1)/numSamples)
    #print 'Fancy new metric (scaled): {}'.format(avg_hamming)

    #if (len(des2) < 30):
    #    avg_hamming = 900000

    #if show == True:
    #    print avg_hamming
    #    img3 = drawMatches(img1, kp1, img2, kp2, matches[:])
    # END ROB

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


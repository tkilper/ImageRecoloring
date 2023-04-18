# CS 440 Project 4
# Tristan Kilper (twk28)

import cv2
import numpy as np
from numpy.core.numeric import Inf

# BASICAGENT
class BasicAgent:

    # FINDREPCOLORS: finds the k representative colors for the input image using k-means clustering.
    @staticmethod
    def findRepColors(image, k):
        
        centers = [] # store centers
        sha = image.shape # store width and height of image
        w = sha[1]
        h = sha[0]
        tol = 10 # tolerance for convergence of clusters
        it = 0 # set iteration counter
        N = 70 # set maximum iteration count

        # randomly assign k centers
        count = 0
        while count < k:
            alrexs = False
            i = np.random.randint(h)
            j = np.random.randint(w)
            tmp = image[i,j]
            for cen in centers:
                if np.array_equal(tmp,cen):
                    alrexs = True
            if not alrexs:
                centers.append(tmp)
                count = count + 1

        # run clustering algorithm
        delta = tol+100 # repeat until convergence
        clusters = np.zeros((h,w))
        while delta > tol and it < N:
            print(centers)
            # assign all pixels to a cluster
            for i in range(h):
                for j in range(w):
                    ptr = image[i,j]
                    mind = Inf
                    ind = 0
                    asicen = ind
                    for cen in centers:
                        tmpd = np.sqrt((int(cen[0])-int(ptr[0]))**2+(int(cen[1])-int(ptr[1]))**2+(int(cen[2])-int(ptr[2]))**2)
                        if tmpd < mind:
                            asicen = ind
                            mind = tmpd
                        ind = ind + 1
                            
                    clusters[i,j] = asicen

            # find new centers and calculator change
            delta = 0
            curclus = []
            for i in range(len(centers)):
                for j in range(h):
                    for k in range(w):
                        if i == clusters[j,k]:
                            curclus.append(image[j,k])
                if len(curclus) == 0:
                    continue
                av = [0,0,0]
                for col in curclus:
                    av = av + col
                av = av / len(curclus)
                delta = delta + np.sum(np.abs(centers[i]-av))
                centers[i] = av
                curclus.clear()

            print(delta)

            # increment iteration counter
            it = it + 1

        # print number of iterations needed and results
        print("number of iterations needed: " + str(it))
        return centers


    # RECOLOR: recolors input image according to the specification of the basic agent in the project
    # description. Left side colored with representative colors and right side done with the algorithm
    # from the description.
    @staticmethod
    def recolor(imagec, imagebw, repcolors):

        sha = imagec.shape # store width and height of image
        w = sha[1]
        h = sha[0]
        mid = int(w / 2) # store midpoint of image

        # extract a b/w left and right half, and a colored left half
        leftc = imagec[0:h-1,0:mid]
        rightc = imagec[0:h,mid+1:w-1]
        leftbw = imagebw[0:h-1,0:mid]
        rightbw = imagebw[0:h,mid+1:w-1]

        # color colored left half with representative colors
        shal = leftc.shape # store width and height of image
        wl = shal[1]
        hl = shal[0]
        for i in range(hl):
            for j in range(wl):
                ptr = leftc[i,j]
                mind = Inf
                newcol = repcolors[0]
                for col in repcolors:
                    tmpd = np.sqrt((int(col[0])-int(ptr[0]))**2+(int(col[1])-int(ptr[1]))**2+(int(col[2])-int(ptr[2]))**2)
                    if tmpd < mind:
                        newcol = col
                        mind = tmpd
                leftc[i,j] = newcol

        # color b/w right half with basic agent algorithm
        shar = rightbw.shape # store width and height of image
        wr = shar[1]
        hr = shar[0]
        for i in range(hr):
            for j in range(wr):
                if i == 0 or j == 0 or i == hr-1 or j == wr-1:
                    rightbw[i,j] = [0,0,0]
                else:
                    testptr = np.array([[rightbw[i-1,j-1],rightbw[i-1,j],rightbw[i-1,j+1]],[rightbw[i,j-1],rightbw[i,j],rightbw[i,j+1]],[rightbw[i+1,j-1],rightbw[i+1,j],rightbw[i+1,j+1]]])
                    simpats = []
                    # find 6 most similar patches from training data
                    potcols = []
                    for k in range(1,hl-1):
                        for m in range(1,wl-1):
                            traiptr = np.array([[leftbw[k-1,m-1],leftbw[k-1,m],leftbw[k-1,m+1]],[leftbw[k,m-1],leftbw[k,m],leftbw[k,m+1]],[leftbw[k+1,m-1],leftbw[k+1,m],leftbw[k+1,m+1]]])
                            if len(simpats) < 6:
                                simpats.append(traiptr)
                                potcols.append(leftc[k,m])
                            else:
                                difftba = np.sum(np.abs(testptr-traiptr))
                                ind = 0
                                for pat in simpats:
                                    diffpat = np.sum(np.abs(testptr-pat))
                                    if difftba < diffpat:
                                        simpats.pop(ind)
                                        potcols.pop(ind)
                                        simpats.append(traiptr)
                                        potcols.append(leftc[k,m])
                                        break
                                    ind = ind + 1
                    # find new color for center
                    counts = [0,0,0,0,0,0]
                    for n in range(len(potcols)):
                        for col in potcols:
                            if np.array_equal(col,potcols[n]):
                                counts[n] = counts[n] + 1
                    highs = [0] # stores potcols index of majority colors
                    for q in range(len(potcols)):
                        if counts[q] > counts[highs[0]]:
                            highs.clear()
                            highs.append(q)
                        if counts[q] == counts[highs[0]]:
                            highs.append(q)
                    
                    if len(highs) > 1:
                        # if there is no majority, color with most similar patch
                        mind = Inf
                        ind = 0
                        newcol = potcols[highs[0]]
                        for pat in simpats:
                            tmpd = np.sum(np.abs(testptr-pat))
                            if tmpd < mind:
                                newcol = potcols[ind]
                                mind = tmpd
                            ind = ind + 1
                        rightbw[i,j] = newcol
                    else:
                        rightbw[i,j] = potcols[highs[0]]

        # return color image, left half colored with rep colors, and right half colored with algorithm
        return imagec, rightbw, leftc
# CS 440 Project 4
# Tristan Kilper (twk28)

from os import stat
import cv2
import numpy as np
import time as t

# IMPROVED AGENT
class ImprovedAgent:

    # SIGMOID: sigmoid function
    @staticmethod
    def sigmoid(x):
        return 1 / (1+np.e**(-x))

    # SIGMOIDPRIME: first derivative of the sigmoid function
    @staticmethod
    def sigmoidprime(x):
        return np.e**(-x) / ((1+np.e**(-x))**2)

    # TRAINNETWORK: take training data
    @staticmethod
    def trainNetwork(self, image, lr, color):

        wt = np.random.normal(0, 0.01, (1,10)) # initialize weights on normal distribution centered at 0
        wt = wt[0]
        tol = 0.0001 # tolerance for convergence of clusters
        sha = image.shape # store width and height of image
        w = sha[1]
        h = sha[0]
        mid = int(w / 2) # store midpoint of image

        # extract the b/w and colored left halves as training data
        leftc = image[0:h-1,0:mid]
        leftbw = cv2.cvtColor(leftc, cv2.COLOR_BGR2GRAY)
        shal = leftc.shape # store width and height of left half
        wl = shal[1]
        hl = shal[0]

        # specify color that will be predicted
        colid = 0
        if color == 'blue':
            colid = 1
        elif color == 'green':
            colid = 2
        
        # train parameters until convergence
        diff = tol+100
        while diff > tol:
            print(wt)
            # compute predicted function value
            i = np.random.randint(1,hl-1)
            j = np.random.randint(1,wl-1)
            input = np.array([1,leftbw[i-1,j-1],leftbw[i-1,j],leftbw[i-1,j+1],leftbw[i,j-1],leftbw[i,j],leftbw[i,j+1],leftbw[i+1,j-1],leftbw[i+1,j],leftbw[i+1,j+1]])
            print(input)
            dot = np.dot(wt,input)
            print(dot)
            ypred = self.sigmoid(dot) * 255
            print('ypred: ' + str(ypred))
            t.sleep(5)
            yr = leftc[i,j][colid]
            # calculate new weights
            for k in range(len(wt)):
                grad = 2*(ypred-yr)*255*self.sigmoidprime(dot)*input[k]
                print(grad)
                t.sleep(5)
                delta = lr * grad
                wt[k] = wt[k] - delta
            delta = delta + np.abs(yr-ypred)

        # return final weights
        return wt

    # RECOLOR: recolor the b/w right half of the image using improved agent model
    def recolor(self, imagebw, redws, bluews, greenws):
        
        sha = imagebw.shape # store width and height of image
        w = sha[1]
        h = sha[0]
        mid = int(w / 2) # store midpoint of image

        # extract the b/w and colored left halves as training data
        rightbw = imagebw[0:h,mid+1:w-1]

        # proceed recolor
        shar = rightbw.shape # store width and height of image
        wr = shar[1]
        hr = shar[0]
        for i in range(hr):
            for j in range(wr):
                if i == 0 or j == 0 or i == hr-1 or j == wr-1:
                    rightbw[i,j] = [0,0,0]
                else:
                    input = np.array([1,rightbw[i-1,j-1],rightbw[i-1,j],rightbw[i-1,j+1],rightbw[i,j-1],rightbw[i,j],rightbw[i,j+1],rightbw[i+1,j-1],rightbw[i+1,j],rightbw[i+1,j+1]])
                    newr = np.dot(redws,input)
                    newr = self.sigmoid(newr[0][0]) * 255
                    newb = np.dot(redws,input)
                    newb = self.sigmoid(newr[0][1]) * 255
                    newg = np.dot(redws,input)
                    newg = self.sigmoid(newr[0][2]) * 255
                    rightbw[i,j][0] = newr
                    rightbw[i,j][1] = newb
                    rightbw[i,j][2] = newg

        # return recolored image
        return rightbw
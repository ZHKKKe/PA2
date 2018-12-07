# CS5487 demo script for Programming Assignment 2
#
# The script has been tested with python 2.7.6
#
# It requires the following modules:
#   numpy 1.8.1
#   matplotlib v1.3.1
#   scipy 0.14.0
#   Image (python image library)

from third_party import pa2
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
# import pylab as pl
from PIL import Image
import scipy.io as sio

from src.algorithm import kmean, mean_shift, em_gmm

def demo():
    import scipy.cluster.vq as vq

    ## load and show image
    img = Image.open('../data/PA2-cluster-images/images/56028.jpg')
    # img = Image.open('../data/PA2-cluster-images/images/101087.jpg')
    pl.subplot(1,3,1)
    pl.imshow(img)
    
    ## extract features from image (step size = 7)
    X,L = pa2.getfeatures(img, 7)
    ## Call kmeans function in scipy.  You need to write this yourself!
    # print(X)
    XT = X.T
    WXT = vq.whiten(XT)

    XTX = WXT.T

    # ------------------- algorithm here
    C,Y = kmean(vq.whiten(XTX), 6, 50, problem_id=2)
    # C,Y = mean_shift(vq.whiten(XTX), [1, 0.25], 1, problem_id=2, visual=False)
    # C,Y = em_gmm(vq.whiten(XTX), 6, problem_id=2, visual=False)

    # C,Y = vqZ.kmeans2(vq.whiten(X.T), 3, iter=1000, minit='random')
    Y = Y + 1 # Use matlab 1-index labeling
    # print(C)
    ## 

    # make segmentation image from labels
    segm = pa2.labels2seg(Y,L)
    pl.subplot(1,3,2)
    pl.imshow(segm)
    
    # color the segmentation image
    csegm = pa2.colorsegms(segm, img)
    pl.subplot(1,3,3)
    pl.imshow(csegm)
    pl.show()

def main():
    demo()
if __name__ == '__main__':
    main()

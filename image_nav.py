import cv2
import numpy as np
#from numpy import arctan2, fliplr, flipu
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import draw

# Read image
img_dir = 'map/keas1.png'
im = cv2.imread(img_dir)

def vis_hist(hist, image, cw, ch, signed_orientation=False):
    w = int(image.shape[1]/cw)
    h = int(image.shape[0]/ch)
    dim = (h,w)
    hist_array = np.zeros(dim) #divide array into subset of image based on cell size

    if signed_orientation:
        max_angle = 2*np.pi
    else:
        max_angle = np.pi

    csx = cw
    csy = ch
    n_cells_y = h
    n_cells_x = w
    nbins = 9;
    sx, sy = n_cells_x*csx, n_cells_y*csy
    center = csx//2, csy//2
    b_step = max_angle / nbins

    radius = min(csx, csy) // 2 - 1
    hog_image = np.zeros((sy, sx), dtype=float)

    for y in range(n_cells_y):
        for x in range(n_cells_x):
            for o in range(nbins):
                centre = tuple([y * csy + csy // 2, x * csx + csx // 2])
                dx = radius * np.cos(o*nbins)
                dy = radius * np.sin(o*nbins)
                rr, cc = draw.line(int(centre[0] - dy),
                                   int(centre[1] - dx),
                                   int(centre[0] + dy),
                                   int(centre[1] + dx))
                hog_image[rr, cc] += hist[y+ x + o]
    return hog_image



def shrink_image(scale, image):
    width = int(image.shape[1]*scale/100)
    height = int(image.shape[0]*scale/100)
    print("scaling image width from " + str(image.shape[1]) + " to " + str(width))
    print("scaling image height from " + str(image.shape[0]) + " to " + str(height))
    dim = (width, height)
    im_resized = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
    return im_resized



image = cv2.imread(img_dir,0)
winSize = (64,64)
blockSize = (64,64)
blockStride = (32,32)
cellSize = (32,32)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
#compute(img[, winStride[, padding[, locations]]]) -> descriptors
winStride = (8,8)
padding = (8,8)
locations = ((10,20),)
hist = hog.compute(image)#,winStride,padding,locations)

print("histogram computed")
print("type of hist is: " + str(type(hist)))
print("size of hist is: " + str(hist.size))
print("shape of hist is: " + str(hist.shape))


hog_im = vis_hist(hist, image, 32, 32)

hog_im = shrink_image(25,hog_im)
cv2.imshow("image gradients",hog_im)
cv2.waitKey(0)

import cv2
import numpy as np
#from numpy import arctan2, fliplr, flipu
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import draw, data, exposure



def shrink_image(scale, image):
    width = int(image.shape[1]*scale/100)
    height = int(image.shape[0]*scale/100)
    print("scaling image width from " + str(image.shape[1]) + " to " + str(width))
    print("scaling image height from " + str(image.shape[0]) + " to " + str(height))
    dim = (width, height)
    im_resized = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
    return im_resized

def vis_hist(hist, image, cw, ch, signed_orientation=False):
    w = int(image.shape[1]/cw)
    h = int(image.shape[0]/ch)
    print h
    print w
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
    hog_image = np.zeros((sy,sx), dtype=float)
    print hog_image.shape
    i=0
    for y in range(n_cells_y):
        i = i+1
        for x in range(n_cells_x):
            for o in range(nbins):
                centre = tuple([y * csy + csy // 2, x * csx + csx // 2])
                dx = radius * np.cos(o*nbins)
                dy = radius * np.sin(o*nbins)
                rr, cc = draw.line(int(centre[0] - dy),
                                   int(centre[1] - dx),
                                   int(centre[0] + dy),
                                   int(centre[1] + dx))
                hog_image[rr,cc] += hist[x,y,o]
    print i
    return hog_image

def ord_hist(hist, nbins):
        ar = np.zeros((218,160,nbins))
        for y in range(0,34662,218):
            for x in range(218):
                for o in range(nbins):
                    ri = y/218
                    ar[x,ri,o] = hist[x+y+o]
        return ar

# def ord_hist(hist, nbins):
#         ar = np.zeros((218,160,nbins))
#         for x in range(0,34720,160):
#             for y in range(160):
#                 for o in range(nbins):
#                     ri = x/160
#                     ar[ri,y,o] = hist[x+y+o]
#         return ar

if __name__ == "__main__":

    # Read image
    img_dir = 'map/keas1.png'
    im = cv2.imread(img_dir)
    # im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # fd = hog(im, orientations=9, pixels_per_cell=(32,32), cells_per_block=(4,4), visualize=False, multichannel=False)

    #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    # hog_im = vis_hist(fd, im, 32, 32)
    # hog_im = shrink_image(35,hog_im)
    # cv2.imshow('hog 2',hog_im)
    # cv2.waitKey(0)

    image = cv2.imread(img_dir,0)
    image = image[:,0:3488]
    winSize = (16,16)
    blockSize = (16,16)
    blockStride = (16,16)
    cellSize = (16,16)
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

    hog_im = ord_hist(hist,nbins)
    hog_im = vis_hist(hog_im, image, 16, 16)
    #hog_im = exposure.rescale_intensity(hog_im, in_range=(0, 100))
    hog_im = shrink_image(45,hog_im)
    cv2.imshow("image gradients",hog_im)
    cv2.waitKey(0)

    #cv2.destroyAllWindows()

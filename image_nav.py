import cv2
import numpy as np
#from numpy import arctan2, fliplr, flipu
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure






# Read image
img_dir = 'map/keas1.png'
im = cv2.imread(img_dir)


# fd, hog_image = hog(im, orientations=9, pixels_per_cell=(64, 64),
#                     cells_per_block=(10, 10), visualize=True, multichannel=True)
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
#
# ax1.axis('off')
# ax1.imshow(im, cmap=plt.cm.gray)
# ax1.set_title('Input image')
#
# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#
# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()

def compute_hist(hist, image, cw, ch):
    w = int(image.shape[1]/cw)
    h = int(image.shape[0]/ch)
    dim = (h,w)
    hist_array = np.zeros(dim) #divide array into subset of image based on cell size

    for x in range(0, w-8, cw):

        for y in range(0, h-8, ch):

            for xx in range(0, 8):
                for yy in range(0, 8):
                    hist_array[y,x] = hist_array[y,x]+ hist[x+y+xx+yy]
    return hist_array



def shrink_image(scale, image):
    width = int(image.shape[1]*scale/100)
    height = int(image.shape[0]*scale/100)
    print("scaling image width from " + str(image.shape[1]) + " to " + str(width))
    print("scaling image height from " + str(image.shape[0]) + " to " + str(height))
    dim = (width, height)
    im_resized = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
    return im_resized

#cv2.imshow("resized raw image",im_resized)
#cv2.waitKey(0)

im = np.float32(im) / 255.0
# Calculate gradient
gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)

# Python Calculate gradient magnitude and direction ( in degrees )
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)


image = cv2.imread(img_dir,0)
winSize = (100,100)
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
hist = hog.compute(image,winStride,padding,locations)

print("histogram computed")
print("type of hist is: " + str(type(hist)))
print("size hist is: " + str(hist.size))
print("shape of hist is: " + str(hist.shape))


nd_hist = compute_hist(hist, image, 8, 8)

gradient_mag = shrink_image(25,mag)

#cv2.imshow("image gradients",nd_hist)
#cv2.waitKey(0)

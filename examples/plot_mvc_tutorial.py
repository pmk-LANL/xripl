"""
Radiograph segmentation tutorial
===================================


"""


#%%
#Importing modules

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median, rank

#from xripl.data import shot
from xripl.reader import openRadiograph
from xripl.contrast import equalize
from xripl.clean import cleanArtifacts, flatten
from xripl.segmentation import detectShock
from skimage.morphology import disk, square
# import xripl.pltDefaults

from xripl.datasets import shot81431


#%%
# Analysis parameters

# image cropping parameters
xMin = 200
xMax = 1800
yMin = 550
yMax = 1400

# spatial calibration 
pxPosReference = (292, 958) # units of px
umPosReference = (1455, 0) # units of um
pxToUm = 0.832 # conversion factor um/px

# indices for extracting shock contour
shockIdxLwr = 335
shockIdxUpr = 890


#%%
# Open the radiograph and plot raw data

# load image data
foreground, background = shot81431()
# flip image
foreground = np.fliplr(foreground)
# plot image and reference point
plt.figure(figsize=(10,12))
plt.imshow(foreground)
plt.scatter(pxPosReference[0], pxPosReference[1], marker="+", c="C6", s=1000)
plt.title(f"Foreground")
plt.show()





#%%

# Cropping the image to the region of interest
img_crop = foreground[yMin:yMax, xMin:xMax]
# normalizing image
img_norm = img_crop / np.max(img_crop)


# transforming coordinates for cropped image
pxRefCrop = (pxPosReference[0] - xMin, pxPosReference[1] - yMin)
plt.imshow(img_crop)
plt.scatter(pxRefCrop[0],
            pxRefCrop[1],
            marker="+",
            c="C6",
            s=1000)
plt.show()


# median filtering to denoise the image
denoised = median(img_norm, disk(5))
denoised_norm = denoised / np.max(denoised)

# flipping image to for more space to plot and to be consistent
# with Ranjan SBI diagram with shock propagating downward.
# these transformations are like a 90 degree rotation counter clockwise
# from the initial image of the shock propagating to the left.
denoised_norm = np.flipud(denoised_norm.transpose())

# removing artifacts (e.g. xrfc streaks, hot pixels, etc.)
cleaned = cleanArtifacts(image=denoised_norm, diskSize=5, plots=True)
# Flattening image by dividing out approxiamte background
flattened = flatten(image=cleaned, medianDisk=10, gaussSize=50, plots=True)


#%% plot flattened image

# transforming coordinates for rotated image
cropShape = np.shape(flattened)
pxRefRot = (pxRefCrop[1], cropShape[0] - pxRefCrop[0])

plt.imshow(flattened)
plt.scatter(pxRefRot[0],
            pxRefRot[1],
            marker="+",
            c="C6",
            s=1000)
plt.show()

#%% Detecting shock, essentially same code as in detectShock()

# shock detection parameters (integer values only)

# We use additional median filtering to help further smooth the image so
# that we capture larger gradient structures instead of small artifacts.
medianDisk = 24
# Radius of the disk kernel used to obtain 2D gradients from the image.
gradientDiskMarkers = 7
# Sets the upper limit of gradients to be used for setting the markers. We
# want to initialize the markers in relatively flat regions.
gradientThreshold = 3
# This parameter mainly affects the visualization of the "local gradient"
# but doesn't really affect the final segmentation much
gradientDiskWatershed = 10

# Watershed segmenting the image using markers set in regions of low
# gradients to initialize the watershed.
shockStuff = detectShock(image=flattened,
                         originalImage=flattened,
                         medianDisk=medianDisk,
                         gradientDiskMarkers=gradientDiskMarkers,
                         gradientThreshold=gradientThreshold,
                         gradientDiskWatershed=gradientDiskWatershed,
                         compactness=1,
                         plots=True,
                         nSegs=10)

# unpacking the results from the segmentation
labelsGrad, contoursGrad, gradientGrad, markersGrad = shockStuff

#%% Remove small regions

from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential

labelsGradLarge = remove_small_objects(labelsGrad, min_size=1000)
labelsGradLargeSeq, _, _ = relabel_sequential(labelsGradLarge, offset=1)

fig, ax = plt.subplots(figsize=(12, 8))
plt.imshow(flattened, cmap=plt.cm.gray, vmin=0.2, vmax=1.0)
plt.imshow(labelsGradLargeSeq, cmap=plt.cm.tab20, alpha=0.3)
plt.title('Segments')
plt.show()

#%% Combining multiple segments on shot 81431 XRFC5

from xripl.segmentation import merge_labels, nSegments, segmentContour


# merging labeled regions that are over-segmented to form a larger region
# that defines the shock
newLabelImg = merge_labels(labels_image=labelsGradLargeSeq,
                           labels_to_merge=[45, 63, 66, 67, 68, 69],
                           label_after_merge=45)

# getting contours for largest area segments
contoursList = []
largestSegments = nSegments(labels=newLabelImg, n=10)
for segmentNumber in largestSegments:
    # get longest contour for each segment
    contourLongest = segmentContour(labels=newLabelImg,
                                    segmentNumber=segmentNumber,
                                    plots=False)
    # save the contour to list
    contoursList.append(contourLongest)

# replacing the original contours with the updated ones
contoursGrad = contoursList

#%% We can select a specific segment label and see which region of the
# original image it corresponds to.
fig, ax = plt.subplots(figsize=(12, 8))
plt.imshow(flattened, cmap=plt.cm.gray, vmin=0.2, vmax=1.0)
plt.imshow(labelsGradLargeSeq == 45, alpha=0.3)
plt.title('Segments')
plt.show()

#%% plots
fig, ax = plt.subplots(figsize=(12, 8))
plt.imshow(flattened, cmap=plt.cm.gray, vmin=0.2, vmax=1.0)
plt.imshow(labelsGradLargeSeq, cmap=plt.cm.tab20, alpha=0.3)
plt.title('Segments')
plt.show()


fig, ax = plt.subplots(figsize=(12, 8))
plt.imshow(flattened, cmap=plt.cm.gray, vmin=0.2, vmax=1.0)
for contour in contoursGrad:
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
plt.title('Contours of segments')
plt.show()


#%% plot of radiographs with reduced contours

# pick index corresponding to shock contour and chop it down
cIdx = 1
shockIdxLwr = 2160
shockIdxUpr = 2850
contourShock = contoursGrad[cIdx][shockIdxLwr:shockIdxUpr, :]
# applying spatial calibrations and offsets to contours and centroid
xContourShock = (contourShock[:, 1] - pxRefRot[0]) * pxToUm + umPosReference[1]
yContourShock = (contourShock[:, 0] - pxRefRot[1]) * pxToUm + umPosReference[0]

cIdx = 0
flatShape = np.shape(flattened)
fig, ax = plt.subplots(figsize=(12, 8))
#xcMin = 0
#xcMax = -1
# getting spatially calibrated axes
hoMin = (0 - pxRefRot[0]) * pxToUm + umPosReference[1]
hoMax = (flatShape[1] - pxRefRot[0]) * pxToUm + umPosReference[1]
vertMin = (0 - pxRefRot[1]) * pxToUm + umPosReference[0]
vertMax = (flatShape[0] - pxRefRot[1]) * pxToUm + umPosReference[0]
extent = [hoMin, hoMax, vertMax, vertMin]

plt.imshow(flattened,
           cmap=plt.cm.gray,
           vmin=0.2,
           vmax=1.0,
           extent=extent,
           origin='upper',)
ax.axis([-300, 300, 1500, 400])
plt.ylabel("Axial (um)")
plt.xlabel("Radial (um)")

# indices for slicing shock contour
#shockIdxLwr = 420
#shockIdxUpr = 950
    
# pick index corresponding to shock contour and chop it down
contourShock = contoursGrad[cIdx][shockIdxLwr:shockIdxUpr, :]
plt.plot(xContourShock, yContourShock, linewidth=2)
plt.show()
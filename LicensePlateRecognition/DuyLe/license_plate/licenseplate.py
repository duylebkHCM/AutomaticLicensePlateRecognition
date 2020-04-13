from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np
import cv2 as cv
import imutils

#define the named tupled to store the license plate
LicensePlate = namedtuple("LicensePlateRegion", ["success", "plate", "thresh", "candidates"])

class License_Plate_Detector:
    def __init__(self, image, minPlateW=60, minPlateH=20, numChars = 7, minCharW = 40):
        self.image = image
        self.minPlateW=minPlateW
        self.minPlateH = minPlateH
        self.numChars = numChars
        self.minCharW = minCharW

    def detect(self):
        lpRegions = self.detectPlates()

        for lpRegion in lpRegions:
            lp = self.detectCharacterCandidates(lpRegion)

            if lp.success is True:
                chars = self.scissor(lp)
                yield (lp, chars)

    def detectPlates(self):
        rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (13,5))
        squareKernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
        regions = []

        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, rectKernel)

        light = cv.morphologyEx(gray, cv.MORPH_CLOSE, squareKernel)
        light = cv.threshold(light, 50, 255, cv.THRESH_BINARY)[1]

        gradX = cv.Sobel(blackhat,  ddepth = cv.cv.CV_32F if imutils.is_cv2() else cv.CV_32F, dx = 1, dy = 0, ksize = -1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255*((gradX - minVal)/(maxVal - minVal))).astype("uint8")

        gradX = cv.GaussianBlur(gradX, (5,5), 0)
        gradX = cv.morphologyEx(gradX, cv.MORPH_CLOSE, rectKernel)
        thresh = cv.threshold(gradX, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

        thresh = cv.erode(thresh, None, iterations=2)
        thresh = cv.dilate(thresh, None, iterations=2)

        thresh = cv.bitwise_and(thresh, thresh, mask=light)
        thresh = cv.dilate(thresh, None, iterations=2)
        thresh = cv.erode(thresh, None, iterations=1)

        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            (w,h) = cv.boundingRect(c)[2:]
            aspectRatio = w / float(h)

            shapeArea = cv.contourArea(c)
            bboxArea = w*h
            extent = shapeArea / float(bboxArea)
            extent = int(extent*100)/100

            rect = cv.minAreaRect(c)
            box = np.int0(cv.cv.BoxPoints(rect)) if imutils.is_cv2() else cv.boxPoints(rect)

            if (aspectRatio > 3 and aspectRatio < 6) and h >  self.minPlateH and w > self.minPlateW and extent > 0.5:
                regions.append(box)
        
        return regions


    def detectCharacterCandidates(self, region):
        plate = perspective.four_point_transform(self.image, region)
        #extract the Value component from the HSV color space and apply adaptive thresholding
        # to reveal the charater on the license plate
        V = cv.split(cv.cvtColor(plate, cv.COLOR_BGR2HSV))[2]
        T = threshold_local(V, 29, offset=15, method="gaussian")
        thresh = (V>T).astype("uint8")*255
        thresh = cv.bitwise_not(thresh)
        
        #resize the license plate region to a canonical size
        plate = imutils.resize(plate, width=400)
        thresh = imutils.resize(thresh, width=400)
        # cv.imshow("Thresh", thresh)

        labels = measure.label(thresh, neighbors=8, background=0)
        charCandidates = np.zeros(thresh.shape, dtype = 'uint8')
        for label in np.unique(labels):
            if label == 0:
                continue

            labelMask = np.zeros(thresh.shape, dtype = 'uint8')
            labelMask[labels == label] = 255
            cnts = cv.findContours(labelMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            if len(cnts) > 0:
                c = max(cnts, key = cv.contourArea)
                (boxX, boxY, boxW, boxH) = cv.boundingRect(c)
                aspectRatio = boxW / float(boxH)
                solidity = cv.contourArea(c) / float(boxW * boxH)
                heightRatio = boxH / float(plate.shape[0])
                keepAspectRatio = aspectRatio < 1.0
                keepSolidity = solidity > 0.15
                keepHeight = heightRatio > 0.4 and heightRatio < 0.95
                # check to see if the component passes all the tests
                if keepAspectRatio and keepSolidity and keepHeight:
                    # compute the convex hull of the contour and draw it on the character
                    # candidates mask
                    hull = cv.convexHull(c)
                    cv.drawContours(charCandidates, [hull], -1, 255, -1)
            
        #TODO
        charCandidates = segmentation.clear_border(charCandidates)
        cnts = cv.findContours(charCandidates.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cv.imshow('Original Image', charCandidates)

        if len(cnts) > self.numChars:
            (charCandidates, cnts) = self.pruneCandidates(charCandidates, cnts)
            cv.imshow('Pruned Candidates', charCandidates)

            # take bitwise AND of raw thresholded image and character candidates to get a more
            # clean segmentation of the characters
        thresh = cv.bitwise_and(thresh, thresh, mask = charCandidates)
        cv.imshow('Char Threshold', thresh)
        return LicensePlate(success=True, plate=plate, thresh=thresh,candidates=charCandidates)

    def pruneCandidates(self, charCandidates, cnts):
        prunedCandidates = np.zeros(charCandidates.shape, dtype = 'uint8')
        dims = []

        for c in cnts:
            (boxX, boxY, boxM, boxH) = cv.boundingRect(c)
            dims.append(boxY+boxH)
            dims = np.array(dims)
            diffs = []
            selected = []

            for i in range(0, len(dims)):
                # compute the sum of differences between the current dimension and and all other
                # dimensions, then update the differences list
                diffs.append(np.absolute(dims - dims[i]).sum())

            # find the top number of candidates with the most similar dimensions and loop over
            # the selected contours
            for i in np.argsort(diffs)[:self.numChars]:
                # draw the contour on the pruned candidates mask and add it to the list of selected
                # contours
                cv.drawContours(prunedCandidates, [cnts[i]], -1, 255, -1)
                selected.append(cnts[i])
                # return a tuple of the pruned candidates mask and selected contours
                return (prunedCandidates, selected)

    def scissor(self, lp):
        # detect contours in the candidates and initialize the list of bounding boxes and
        # list of extracted characters
        cnts = cv.findContours(lp.candidates.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        boxes = []
        chars = []

        for c in cnts:
            # compute the bounding box for the contour while maintaining the minimum width
            (boxX, boxY, boxW, boxH) = cv.boundingRect(c)
            dX = min(self.minCharW, self.minCharW - boxW) // 2
            boxX -= dX
            boxW += (dX * 2)
            # update the list of bounding boxes
            boxes.append((boxX, boxY, boxX + boxW, boxY + boxH))
        
        # sort the bounding boxes from left to right
        boxes = sorted(boxes, key=lambda b:b[0])

        # loop over the started bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # extract the ROI from the thresholded license plate and update the characters
            # list
            chars.append(lp.thresh[startY:endY, startX:endX])
        # return the list of characters
        return chars
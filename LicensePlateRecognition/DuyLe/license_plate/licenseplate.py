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
                yield (lp, lpRegion)

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
        cv.imshow("Perspective Transform", imutils.resize(plate, width=400))
        #extract the Value component from the HSV color space and apply adaptive thresholding
        # to reveal the charater on the license plate
        V = cv.split(cv.cvtColor(self.image, cv.COLOR_BGR2HSV))
        T = threshold_local(V, 29, offset=15, method="gaussian")
        thresh = (V>T).astype("uint8")*255
        thresh = cv.bitwise_not(thresh)
        
        #resize the license plate region to a canonical size
        plate = imutils.resize(plate, width=400)
        thresh = imutils.resize(thresh, width=400)
        cv.imshow("Thresh", thresh)

        
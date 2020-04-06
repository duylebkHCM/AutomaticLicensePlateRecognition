from __future__ import print_function
from DuyLe.license_plate.licenseplate import License_Plate_Detector
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2 as cv

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the images to be classified")
args = vars(ap.parse_args())

for imagePath in sorted(list(paths.list_images(args["image"]))):
    image = cv.imread(imagePath)
    print(imagePath)

    if image.shape[1] > 640:
        image = imutils.resize(image, width=640)

    lpd = License_Plate_Detector(image)
    plates = lpd.detect()

    for lpBox in plates:
        print(lpBox)
        lpBox = np.array(lpBox).reshape((-1,1,2)).astype(np.int32)
        print(lpBox)
        cv.drawContours(image, [lpBox], -1, (0,255,0), 2)

    cv.imshow("image", image)
    cv.waitKey(0)
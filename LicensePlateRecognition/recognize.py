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

    # for (i, (lp,lpBox)) in enumerate(plates):
    #     print(lpBox)
    #     lpBox = np.array(lpBox).reshape((-1,1,2)).astype(np.int32)
    #     cv.drawContours(image, [lpBox], -1, (0,255,0), 2)

    #     candidates = np.dstack([lp.candidates]*3)
    #     thresh = np.dstack([lp.thresh]*3)
    #     output = np.vstack([lp.plate, thresh, candidates])
    #     cv.imshow('Plate and Candidate #{}'.format(i+1), output)

    for (lpBox, chars) in plates:
        for (i, char) in enumerate(chars):
            cv.imshow('Character {}'.format(i+1), char)
            
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
import cv2 as cv
import glob
import re

for images in glob.glob("../crater_dataset/crater_data/images/tile3_24/crater/*.jpg"):
    img = cv.imread(images)
    img = cv.resize(img, (200, 200))
    cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)

    img_name = re.split('TE', images)
    img_final = re.split('\.', img_name[1])
    img_dest = '../crater_dataset/crater_data/images/normalized_images/crater/' + 'TE' + img_final[0] + '_norm' + '.jpg'

    cv.imwrite(img_dest, img)

for images in glob.glob("../crater_dataset/crater_data/images/tile3_24/non-crater/*.jpg"):

    img = cv.imread(images)
    img = cv.resize(img, (200, 200))
    cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)

    img_name = re.split('FE', images)
    img_final = re.split('\.', img_name[1])
    img_dest = '../crater_dataset/crater_data/images/normalized_images/non-crater/' + 'FE' + img_final[
        0] + '_norm' + '.jpg'

    cv.imwrite(img_dest, img)


from evaluation.define import Rectangle
import copy
import cv2


def draw_rec_on_img(img, rec, text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    line_type = 2

    upper_left = (rec.xmin, rec.ymin)
    bottom_right = (rec.xmax, rec.ymax)

    cv2.rectangle(img, upper_left, bottom_right, color, 2)
    cv2.putText(img, text, upper_left, font, font_scale, color, line_type)



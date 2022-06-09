import cv2
import numpy as np
from matplotlib import pyplot as plt


## BASIC TRANSFORMATION FUNCTION
# SHOW with openCV
def show_img(img):
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# SHOW with matplotlib
def show_image(img, **kwargs):
    """
    Show an RGB numpy array of an image without any interpolation
    """

    plt.subplot()
    plt.axis('off')
    plt.imshow(
        X=img,
        interpolation='none',
        **kwargs
    )


def apply_border(img):
    img = cv2.copyMakeBorder(
        src=img,
        top=10,
        bottom=10,
        left=10,
        right=10,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255), )

    h, w, c = img.shape
    print(f'Image shape: {h}H x {w}W x {c}C')
    return img


def resize_img(img, MAX_PIX=800):
    """
    Resize an RGB numpy array of an image, either along the height or the width, and keep its aspect ratio. Show restult.
    """
    h, w, c = img.shape

    print('w', w, 'MAX_PIX', MAX_PIX)

    if h > MAX_PIX:
        flag = 'h'

    if w > MAX_PIX:
        flag = 'w'

    if flag == 'h':
        dsize = (int((MAX_PIX * w) / h), int(MAX_PIX))
    else:
        dsize = (int(MAX_PIX), int((MAX_PIX * h) / w))

    img_resized = cv2.resize(
        src=img,
        dsize=dsize,
        interpolation=cv2.INTER_CUBIC,
    )

    h, w, c = img_resized.shape
    print(f'Image shape: {h}H x {w}W x {c}C')

    show_image(img_resized)

    return img_resized

    if h > MAX_PIX:
        img_resized = resize_image(img, 'h')

    if w > MAX_PIX:
        img_resized = resize_image(img, 'w')

    return img


def blur_img(img, k_b=5):
    img = cv2.GaussianBlur(
        src=img,
        ksize=(k_b, k_b),
        sigmaX=0,
        sigmaY=0,
    )
    show_img(img)
    return img

img_o = cv2.imread(filename='simp2.jpg',flags=cv2.IMREAD_COLOR,)

h, w, c = img_o.shape
print(f'Image shape: {h}H x {w}W x {c}C')
img = apply_border(img_o)
img= resize_img(img)
# img = blur_img(img)
# show_img(img)


morph_size = 0
max_operator = 4
max_elem = 2
max_kernel_size = 21
title_trackbar_operator_type = 'Operator:\n 0: Opening - 1: Closing  \n 2: Gradient - 3: Top Hat \n 4: Black Hat'
title_trackbar_element_type = 'Element:\n 0: Rect - 1: Cross - 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n + 1'
title_window = 'test'
morph_op_dic = {0: cv2.MORPH_OPEN, 1: cv2.MORPH_CLOSE, 2: cv2.MORPH_GRADIENT, 3: cv2.MORPH_TOPHAT, 4: cv2.MORPH_BLACKHAT}
el_type_dic = {0:'Rect', 1:'Cross', 2:'Elipse'}

def morphology_operations(val):
    morph_operator = cv2.getTrackbarPos(title_trackbar_operator_type, title_window)
    morph_size = cv2.getTrackbarPos(title_trackbar_kernel_size, title_window)
    morph_elem = 0
    val_type = cv2.getTrackbarPos(title_trackbar_element_type, title_window)
    if val_type == 0:
        morph_elem = cv2.MORPH_RECT
    elif val_type == 1:
        morph_elem = cv2.MORPH_CROSS
    elif val_type == 2:
        morph_elem = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(morph_elem, (2*morph_size + 1, 2*morph_size+1), (morph_size, morph_size))
    operation = morph_op_dic[morph_operator]
    dst = cv2.morphologyEx(img, operation, element)
    print('Val_type: ', el_type_dic[val_type], 'Morph operator:', operation, 'Morph size: ', morph_size)
    cv2.imshow(title_window, dst)


cv2.namedWindow(title_window)
cv2.createTrackbar(title_trackbar_operator_type, title_window , 0, max_operator, morphology_operations)
cv2.createTrackbar(title_trackbar_element_type, title_window , 0, max_elem, morphology_operations)
cv2.createTrackbar(title_trackbar_kernel_size, title_window , 0, max_kernel_size, morphology_operations)

morphology_operations(img)

cv2.waitKey(0)
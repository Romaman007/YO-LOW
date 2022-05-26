import cv2
import cv2 as cv
import numpy as np
from PIL import Image


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def get_anchors(anchors_path):
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


def preprocess_input(image):
    image /= 255.0
    return image


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 3)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def resize(img, size):
    if img.size[0]<size or img.size[1]<size:
        if float(img.size[0]) <= float(img.size[1]):
            width = size
            img = img.convert('RGB')
            ratio = (width / float(img.size[0]))
            height = int((float(img.size[1]) * float(ratio)))
            img = img.resize((width, height), Image.BICUBIC)
        else:
            height = size
            img = img.convert('RGB')
            ratio = (height / float(img.size[1]))
            width = int((float(img.size[0]) * float(ratio)))
            img = img.resize((width, height), Image.BICUBIC)
    img.save('resized_image1.jpg')


def im_cut(image, t, l, b, r):
    output = image.crop((l, t, r, b))
    output.save("Partcoords(%d,%d,%d,%d).png" % (l, t, r, b))
    resize(output, 600)
    img = cv2.imread('resized_image1.jpg')
    return img


def laplacian_sharpening(img, K_size=3):
    H, W = img.shape

    # zero padding

    pad = K_size // 2

    out = np.zeros((H + pad * 2, W + pad * 2))

    out[pad: pad + H, pad: pad + W] = img.copy()

    tmp = out.copy()

    # laplacian kernle

    K = [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]

    # filtering and adding image -> Sharpening image

    for y in range(H):

        for x in range(W):
            # core code

            out[pad + y, pad + x] = (-1) * np.sum(K * (tmp[y: y + K_size, x: x + K_size])) + tmp[pad + y, pad + x]

    out = np.clip(out, 0, 255)

    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out


if __name__ == "__main__":
    # from matplotlib import pyplot as plt
    image = Image.open("test/090.jpg").convert("RGB")

    resize(image,800)
    img = cv.imread("resized_image1.jpg")
    img = get_grayscale(img)
    img2 = cv.GaussianBlur(img, (3, 3), 0)
    img3 = remove_noise(img)
    # cv2.imwrite('1111g.jpg', img2)
    # cv2.imwrite('12233.jpg', img3)
    # cv2.imwrite('1223dd3.jpg', canny(img))
    # cv2.imwrite('122dss33.jpg', erode(img))
    # cv2.imwrite('122dsxxxxs33.jpg', dilate(img))
    # cv2.imwrite('122dssxsdsd33.jpg', opening(img))
    cv2.imwrite('test/122dssxsdcxcxsd334.jpg', laplacian_sharpening(img2))


    #
    # hist, bins = np.histogram(img, 256)
    # # calculate cdf
    # cdf = hist.cumsum()
    # # plot hist
    # plt.plot(hist, 'r')
    #
    # # remap cdf to [0,255]
    # cdf = (cdf - cdf[0]) * 255 / (cdf[-1] - 1)
    # cdf = cdf.astype(np.uint8)  # Transform from float64 back to unit8
    #
    # # generate img after Histogram Equalization
    # img2 = np.zeros((384, 495, 1), dtype=np.uint8)
    # img2 = cdf[img]
    #
    # hist2, bins2 = np.histogram(img2, 256)
    # cdf2 = hist2.cumsum()
    # plt.plot(hist2, 'g')
    #
    #
    # plt.show()
    # print(img2)
    # cv2.imwrite('color_img.jpg', img2)
    # # show img after histogram equalization

import sys
from math import sqrt, atan2, pi
import numpy as np
import imageio
import skimage.color as sk
import matplotlib.pyplot as plt


def canny_edge_detector(input_image):
    input_pixels = input_image.load()
    width = input_image.width
    height = input_image.height

    # Transform the image to grayscale
    grayscaled = compute_grayscale(input_pixels, width, height)

    # Blur it to remove noise
    blurred = compute_blur(grayscaled, width, height)

    # Compute the gradient
    gradient, direction = compute_gradient(blurred, width, height)

    # Non-maximum suppression
    filter_out_non_maximum(gradient, direction, width, height)

    # Filter out some edges
    keep = filter_strong_edges(gradient, width, height, 20, 25)

    return keep


def compute_grayscale(input_pixels, width, height):
    grayscale = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            pixel = input_pixels[x, y]
            grayscale[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3
    return grayscale


def compute_blur(input_pixels, width, height):
    # Keep coordinate inside image
    clip = lambda x, l, u: l if x < l else u if x > u else x

    # Gaussian kernel
    kernel = np.array([
        [1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256]
    ])

    # Middle of the kernel
    offset = len(kernel) // 2

    # Compute the blurred image
    blurred = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            acc = 0
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = clip(x + a - offset, 0, width - 1)
                    yn = clip(y + b - offset, 0, height - 1)
                    acc += input_pixels[xn, yn] * kernel[a, b]
            blurred[x, y] = int(acc)
    return blurred


def compute_gradient(input_pixels, width, height):
    gradient = np.zeros((width, height))
    direction = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            if 0 < x < width - 1 and 0 < y < height - 1:
                magx = input_pixels[x + 1, y] - input_pixels[x - 1, y]
                magy = input_pixels[x, y + 1] - input_pixels[x, y - 1]
                gradient[x, y] = sqrt(magx ** 2 + magy ** 2)
                direction[x, y] = atan2(magy, magx)
    return gradient, direction


def filter_out_non_maximum(gradient, direction, width, height):
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            angle = direction[x, y] if direction[x, y] >= 0 else direction[x, y] + pi
            rangle = round(angle / (pi / 4))
            mag = gradient[x, y]
            if ((rangle == 0 or rangle == 4) and (gradient[x - 1, y] > mag or gradient[x + 1, y] > mag)
                    or (rangle == 1 and (gradient[x - 1, y - 1] > mag or gradient[x + 1, y + 1] > mag))
                    or (rangle == 2 and (gradient[x, y - 1] > mag or gradient[x, y + 1] > mag))
                    or (rangle == 3 and (gradient[x + 1, y - 1] > mag or gradient[x - 1, y + 1] > mag))):
                gradient[x, y] = 0


def filter_strong_edges(gradient, width, height, low, high):
    # Keep strong edges
    keep = set()
    for x in range(width):
        for y in range(height):
            if gradient[x, y] > high:
                keep.add((x, y))

    # Keep weak edges next to a pixel to keep
    lastiter = keep
    while lastiter:
        newkeep = set()
        for x, y in lastiter:
            for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if gradient[x + a, y + b] > low and (x + a, y + b) not in keep:
                    newkeep.add((x + a, y + b))
        keep.update(newkeep)
        lastiter = newkeep

    return list(keep)


def detect_circles(filename):
    from PIL import Image, ImageDraw
    from math import sqrt, pi, cos, sin
    from collections import defaultdict

    # Load image:
    input_image = Image.open(filename)

    # Output image:
    output_image = Image.new("RGB", input_image.size)
    output_image.paste(input_image)
    draw_result = ImageDraw.Draw(output_image)

    # Find circles
    rmin = 18
    rmax = 20
    steps = 100
    threshold = 1

    points = []
    for r in range(rmin, rmax + 1):
        for t in range(steps):
            points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

    acc = defaultdict(int)
    for x, y in canny_edge_detector(input_image):
        for r, dx, dy in points:
            a = x - dx
            b = y - dy
            acc[(a, b, r)] += 1

    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            print(v / steps, x, y, r)
            circles.append((x, y, r))

    for x, y, r in circles:
        draw_result.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0, 0))

    # Save output image
    output_image.save("result.png")


GRAY_REP = 1
RGB_REP = 2
RGB2YIQ_MATRIX = np.array(
    [[0.299, 0.587, 0.114], [0.59590059, -0.27455667, -0.32134392], [0.21153661, -0.52273617, 0.31119955]])
YIQ2RGB_MATRIX = np.array([[1, 0.95569, 0.61986], [1, -0.27158, -0.64687], [1, -1.10818, 1.70506]])
BINS = 256


# returns true is is an rbg image
def is_rgb(img):
    return (img.ndim == 3)


# Receives a filename (grayscale/RGB) and a desired representation-
# 1=grayscale & 2=RGB. The function returns an image of type np.float64 with
# intensities normalized to [1,0].
def read_image(filename, representation):
    img = imageio.imread(filename) / 256  # reads file into array of float64
    if (not is_rgb(img)) or (is_rgb(img) and representation == RGB_REP):
        return img
    return sk.rgb2gray(img)


# receives an image and a representation, and displays it, using matplotlib library
def imdisplay(filename, representation):
    converted = read_image(filename, representation)
    plt.imshow(converted, cmap='gray')
    plt.show()


# receives an RGB image, and transforms it to YIQ color-space
def rgb2yiq(imRGB):
    yiq = imRGB.copy()
    yiq[:, :, 0] = RGB2YIQ_MATRIX[0][0] * imRGB[:, :, 0] + RGB2YIQ_MATRIX[0][1] * imRGB[:, :, 1] + \
                   RGB2YIQ_MATRIX[0][2] * imRGB[:, :, 2]
    yiq[:, :, 1] = RGB2YIQ_MATRIX[1][0] * imRGB[:, :, 0] + RGB2YIQ_MATRIX[1][1] * imRGB[:, :, 1] + \
                   RGB2YIQ_MATRIX[1][2] * imRGB[:, :, 2]
    yiq[:, :, 2] = RGB2YIQ_MATRIX[2][0] * imRGB[:, :, 0] + RGB2YIQ_MATRIX[2][1] * imRGB[:, :, 1] + \
                   RGB2YIQ_MATRIX[2][2] * imRGB[:, :, 2]
    return yiq


# receives a YIQ image, and transforms it to RGB color space
def yiq2rgb(imYIQ):
    rgb = imYIQ.copy()
    rgb[:, :, 0] = YIQ2RGB_MATRIX[0][0] * imYIQ[:, :, 0] + YIQ2RGB_MATRIX[0][1] * imYIQ[:, :, 1] + \
                   YIQ2RGB_MATRIX[0][2] * imYIQ[:, :, 2]
    rgb[:, :, 1] = YIQ2RGB_MATRIX[1][0] * imYIQ[:, :, 0] + YIQ2RGB_MATRIX[1][1] * imYIQ[:, :, 1] + \
                   YIQ2RGB_MATRIX[1][2] * imYIQ[:, :, 2]
    rgb[:, :, 2] = YIQ2RGB_MATRIX[2][0] * imYIQ[:, :, 0] + YIQ2RGB_MATRIX[2][1] * imYIQ[:, :, 1] + \
                   YIQ2RGB_MATRIX[2][2] * imYIQ[:, :, 2]
    return rgb


def plot_histogram(img):
    img = (img * BINS).astype(int)  # mult. by 256 and round to integers

    # calculate histogram and normalized cumulative histogram
    counts, bin_edges = np.histogram(img, BINS, density=True)
    plt.bar(bin_edges[:-1] - 0.5, counts, width=1, edgecolor='none')
    plt.xlim([-0.5, 255.5])
    plt.show()


# receives an RGB/Gray-level image (floats [0,1]), returns a list [im_eq, hist_orig, hist_eq]
# im_eq := the equalized image
# hist_orig := 256 bin histogram of the original image
# hist_eq := 256 bin histogram of the new image
def histogram_equalize(im_orig):
    # prepare image for operations - if rgb make yiq
    if is_rgb(im_orig):
        im_yiq = rgb2yiq(im_orig.copy())
        img = im_yiq[:, :, 0].copy()
    else:
        img = im_orig.copy()
    img = (img * BINS).astype(int)  # mult. by 256 and round to integers

    # calculate histogram and normalized cumulative histogram
    hist_orig, bin_edges = np.histogram(img, BINS, density=True)
    cum_hist = np.cumsum(hist_orig)
    cum_hist = (cum_hist / cum_hist[BINS - 1]) * (BINS - 1)

    # create look up table by the given formula (stretch and round)
    c_m, c_255 = cum_hist[np.nonzero(cum_hist)[0][0]], cum_hist[BINS - 1]
    LUT = (cum_hist - c_m) / (c_255 - c_m) * (BINS - 1)

    # map the intensity values according to the look up table
    if is_rgb(im_orig):
        im_yiq[:, :, 0] = LUT[img] / BINS
        im_eq = yiq2rgb(im_yiq)
    else:
        im_eq = LUT[img] / BINS

    # calculate histogram of the equalized image
    hist_eq = np.histogram(im_eq.copy(), BINS, density=True)[0]

    return [im_eq, hist_orig, hist_eq]


# performs optimal image quantization of RGB/gray-scale image. Parameters:
# im_orig := the image to manipulate
# n_quant := number of intensities for output image
# n_iter := maximum num of iterations for optimization
# Output: a list [im_quant, error]
def quantize(im_orig, n_quant, n_iter):
    # prepare image for operations - if rgb make yiq
    rgb_flag = (im_orig.ndim == 3)
    if rgb_flag:
        im_yiq = rgb2yiq(im_orig.copy())
        img = im_yiq[:, :, 0].copy()
    else:
        img = im_orig.copy()
    img *= BINS

    # initialize s-array, q-array, error array and look-up-table
    z = np.zeros(shape=(n_quant + 1)).astype("float64")
    z[0], z[n_quant] = 0, BINS - 1
    q = np.zeros(shape=n_quant).astype("float64")
    err = np.zeros(shape=n_iter).astype("float64")

    # calculate normalized cumulative histogram
    hist, bin_edges = np.histogram(img, BINS, density=True)
    cum_hist = np.cumsum(hist)

    # find first Zi's: iterate over n_quant, in each step find the next Zi
    delta = cum_hist[BINS - 1] / n_quant
    for i in range(n_quant - 1):
        z[i + 1] = np.where(cum_hist >= delta * (i + 1))[0][0] - 1

    # find first Qi's
    for i in range(n_quant):
        start, end = round(z[i]) + 1, round(z[i + 1])
        q[i] = np.average(np.arange(start, end), weights=hist[int(start):int(end)])

    # LOOP:  find Qi's and then Zi's, until convergence
    for n in range(n_iter):
        # compute new Zi's:
        former_z = z.copy()
        for i in range(1, n_quant):
            z[i] = (q[i - 1] + q[i]) / 2

        # compute new Qi's and error:
        for i in range(1, n_quant):
            start, end = round(z[i]) + 1, round(z[i + 1])
            q[i] = np.average(np.arange(start, end), weights=hist[int(start):int(end)])
            err[n] += (((q[i] - np.arange(start, end)) ** 2) * hist[int(start):int(end)]).sum()

        # check convergence:
        if np.array_equal(former_z, z):
            break

    # iterate and apply the q's on the pixels of the image
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            i = 0
            found = False
            while i < n_quant and (not found):
                if img[r][c] < z[i + 1]:
                    img[r][c] = q[i]
                    found = True
                i += 1

    # convert back to rgb if necessary
    if is_rgb(im_orig):
        im_yiq[:, :, 0] = img / BINS
        im_qua = yiq2rgb(im_yiq)
    else:
        im_qua = img / BINS

    return [im_qua, err]

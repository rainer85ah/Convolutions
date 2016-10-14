import argparse as arg
import numpy as np
from skimage import exposure

import cv2


def convolve(image, kernel):
    """
    :param image: Grayscale image
    :param kernel: Matrix N*M Odd numbers.
    :return: image result after applying the kernel
    """
    (image_height, image_width) = image.shape[:2]  # grab the spatial dimensions of the image.
    (kernel_height, kernel_width) = kernel.shape[:2]  # grab the spatial dimensions of the kernel.

    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # size (i.e., width and height) are not reduced
    pad = (kernel_width - 1) / 2  # useful for an output image with the same size that de original one.
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros_like((image_height, image_width), dtype=np.float32)

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to bottom.
    for y in np.arange(pad, image_height + pad):
        for x in np.arange(pad, image_width + pad):
            # extract the ROI of the image by extracting the *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perform the actual convolution by taking the element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            convolution_value = (roi * kernel).sum()
            # store the convolved value in the output (x,y)- coordinate of the output image
            output[y - pad, x - pad] = convolution_value

    # rescale the output image to be in the range [0, 255]
    output = exposure.rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype(np.uint8)

    # return the output image
    return output


# construct the argument parse and parse the arguments
ap = arg.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

# construct the Laplacian kernel used to detect edges regions of an image
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

# construct the Sobel y-axis kernel
sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")

# construct the kernel bank, a list of kernels we're going to apply using both our custom `convole` function and
# OpenCV's `filter2D` function
kernel_bank = (("small_blur", smallBlur), ("large_blur", largeBlur), ("sharpen", sharpen), ("laplacian", laplacian),
               ("sobel_x", sobelX), ("sobel_y", sobelY))

img = cv2.imread(args["image"])  # load the input image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert the image to grayscale image

# loop over the kernels
for (kernelName, kernel) in kernel_bank:
    # apply the kernel to the grayscale image using both our custom `convole` function and OpenCV's `filter2D` function
    print("[INFO] applying {} kernel".format(kernelName))
    # convole_output = convolve(gray, kernel)
    opencv_output = cv2.filter2D(gray, -1, kernel)

    # show the output images
    cv2.imshow("Color Original", img)
    cv2.imshow("Grayscale Original", gray)
    # cv2.imshow("{} - Convole".format(kernelName), convole_output)
    cv2.imshow("{} - Opencv".format(kernelName), opencv_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

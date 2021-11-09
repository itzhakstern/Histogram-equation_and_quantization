import numpy as np
from imageio import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


MAX_GRAY_COLOR = 256
DIM_OF_GRAY_IMAGE = 2
LEVEL_OF_GRAYSCALE = 255
RGB_TO_YIQ = np.float64([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
YIQ_TO_RGB = np.linalg.inv(RGB_TO_YIQ)


def read_image(filename, representation):
    """
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2).
    :return: This function returns an image, that represented by a matrix of type np.float64 with intensities
    (either grayscale or RGB channel intensities) normalized to the range [0, 1].
    """
    grayScale = 1
    image = imread(filename) / LEVEL_OF_GRAYSCALE
    if image.ndim != DIM_OF_GRAY_IMAGE and representation == grayScale:
        image = rgb2gray(image)
    return np.float64(image)


def imdisplay(filename, representation):
    """
    a function to show image on the screen
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2).
    """
    image = read_image(filename, representation)
    plt.imshow(image)
    plt.show()


def rgb2yiq(imRGB):
    """
    replace a representation of image in RBG to YIQ
    :param imRGB: an image that represented by RBG
    (3 matrix of type np.float64 with intensities normalized to the range [0, 1])
    :return: an image that represented by YIQ
    """
    return imRGB @ RGB_TO_YIQ.T


def yiq2rgb(imYIQ):
    """
    replace a representation of image in YIQ to RGB
    :param imRGB: an image that represented by YIQ
    (3 matrix of type np.float64 with intensities normalized to the range [0, 1])
    :return: an image that represented by RGB
    """
    return imYIQ @ YIQ_TO_RGB.T


def gray_image(im_orig):
    """
    :param im_orig: an image in a RGB or grayscale presentation
    :return: a gray image (if the original image has in RGB presentation we
    replace the presentation to YIQ and return the Y metrix)
    """
    if im_orig.ndim == DIM_OF_GRAY_IMAGE:
        return im_orig.copy(), []
    yiqImage = rgb2yiq(im_orig)
    return yiqImage[:, :, 0], yiqImage


def histogram_equalize(im_orig):
    """
    a function to equalization the histogram of image in order to use of all gray levels and improve the image contrast
    :param im_orig:  is the input grayscale or RGB float64 image with values in [0, 1].
    :return: a list [im_eq, hist_orig, hist_eq] where
            im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
            hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
            hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
    """
    adapted_image, yiqImage = gray_image(im_orig)
    hist_orig, bounds_orig = np.histogram(adapted_image, bins=MAX_GRAY_COLOR, range=(0,1))
    hist_cumulative = np.cumsum(hist_orig)
    c_m = hist_cumulative[np.nonzero(hist_cumulative)[0][0]]
    lookUpTable = np.round(LEVEL_OF_GRAYSCALE * ((hist_cumulative - c_m) / (hist_cumulative[LEVEL_OF_GRAYSCALE] - c_m)))
    temp_im_eq = lookUpTable[(adapted_image * LEVEL_OF_GRAYSCALE).astype(np.int)] / LEVEL_OF_GRAYSCALE
    if not im_orig.ndim == DIM_OF_GRAY_IMAGE:
        yiqImage[:, :, 0] = temp_im_eq
    im_eq = temp_im_eq if im_orig.ndim == DIM_OF_GRAY_IMAGE else yiq2rgb(yiqImage)
    hist_eq, bounds_eq = np.histogram(im_eq, MAX_GRAY_COLOR)
    return [im_eq, hist_orig, hist_eq]


def initial_Z_and_Q_tables(hist_orig, n_quant):
    """
    :param hist_orig: original histogram of image
    :param bounds_orig: array of all gray levels
    :param n_quant: number of colors that we want sae the image
    :return: zTable - array of Boundaries of the division of color levels
             qTable - array of the color truth that will be in each division segment
    """
    qTable = np.zeros(n_quant)
    hist_cumulative = np.cumsum(hist_orig)
    zi_sequence = hist_cumulative[-1] / n_quant
    zTable = [np.argmin(np.abs(hist_cumulative - (zi_sequence * i))) for i in range(1, n_quant)]
    zTable = [-1] + zTable + [255]
    return zTable, qTable


def adapted_Q_table(qTable, zTable, hist_orig, n_quant):
    """
    adapted the q table
    :param qTable: array of the color truth that will be in each division segment
    :param zTable: array of Boundaries of the division of color levels
    :param hist_orig: original histogram of image
    :param n_quant: number of colors that we want sae the image
    :param bounds_orig: array of all gray levels
    """
    all_gray_levels = np.arange(MAX_GRAY_COLOR)
    for i in range(n_quant):
        up_border = np.where(all_gray_levels > zTable[i])[0]
        low_border = np.where(all_gray_levels <= zTable[i + 1])[0]
        idex_array = np.intersect1d(up_border, low_border)

        g_hg = np.dot(idex_array, hist_orig[idex_array])
        hg = np.sum(hist_orig[idex_array])
        qTable[i] = g_hg / hg


def adapted_Z_table(qTable, zTable, n_quant):
    """
    adapted the z table
    :param qTable: array of the color truth that will be in each division segment
    :param zTable: array of Boundaries of the division of color levels
    :param n_quant: number of colors that we want sae the image
    """
    for z in range(1, n_quant):
        zTable[z] = (qTable[z] + qTable[z - 1]) / 2


def adapted_error(zTable, qTable, hist_orig, n_quant, error):
    """
    adapted the error table
    :param zTable: array of Boundaries of the division of color levels
    :param qTable: array of the color truth that will be in each division segment
    :param hist_orig: original histogram of image
    :param n_quant: number of colors that we want sae the image
    :param error: array of errors in calculating the color distribution until convergence
    :param bounds_orig: array of all gray levels
    """
    error_element = 0
    all_gray_levels = np.arange(MAX_GRAY_COLOR)
    for i in range(n_quant):
        up_border = np.where(all_gray_levels > zTable[i])[0]
        low_border = np.where(all_gray_levels <= zTable[i + 1])[0]
        idex_array = np.intersect1d(up_border, low_border)
        error_element += np.dot(np.power(qTable[i] - idex_array, 2), hist_orig[idex_array])
    error.append(error_element)


def quantize(im_orig, n_quant, n_iter):
    """
    :param im_orig: is the input grayscale or RGB image to be quantized (float64 image with values in [0, 1])
    :param n_quant: is the number of intensities your output im_quant image should have
    :param n_iter: is the maximum number of iterations of the optimization procedure (may converge earlier.)
    :return: the output is a list [im_quant, error] where
    im_quant - is the quantized output image. (float64 image with values in [0, 1]).
    error - is an array with shape (n_iter,) (or less) of the total intensities error for each iteration of the
    quantization procedure.
    """
    adapted_image, yiqImage = gray_image(im_orig)
    hist_orig, bounds_orig = np.histogram(adapted_image, bins=MAX_GRAY_COLOR, range=(0,1))
    zTable, qTable = initial_Z_and_Q_tables(hist_orig, n_quant)
    error = []
    temp_z = np.zeros(n_quant + 1)

    # adapted the values of colors (qTable) according the division of all
    # colors levels (zTable) and opposite until the error will convene
    for iter in range(n_iter):
        if np.array_equal(temp_z, zTable):
            break
        temp_z = zTable.copy()

        adapted_Q_table(qTable, zTable, hist_orig, n_quant)
        adapted_Z_table(qTable, zTable, n_quant)
        adapted_error(zTable, qTable, hist_orig, n_quant, error)

    for i in range(1, n_quant + 1):
        adapted_image[np.where((adapted_image >= zTable[i - 1] / LEVEL_OF_GRAYSCALE) &
                               (adapted_image <= zTable[i] / LEVEL_OF_GRAYSCALE))] = qTable[i - 1] / LEVEL_OF_GRAYSCALE

    if not im_orig.ndim == 2:
        yiqImage[:, :, 0] = adapted_image
    im_quant = adapted_image if im_orig.ndim == DIM_OF_GRAY_IMAGE else yiq2rgb(yiqImage)
    return [im_quant, np.array(error[:-1])]


def quantize_rgb(im_orig, n_quant):
    """
    this function make quantization to a RGB image
    (using in "sklearn.cluster" library)
    :param im_orig: is the input RGB image to be quantized
    :param n_quant: is the number of intensities your output im_quant image should have
    :return: im_ready - is the quantized output image.
    """
    from sklearn.cluster import KMeans as quantize_means
    shape = im_orig.shape[:2]
    new_shape = (shape[0] * shape[1], 3)
    im_shaped = np.reshape(im_orig, new_shape)
    kmeans_cluster = quantize_means(n_clusters=n_quant, random_state=0)
    kmeans_fitted = kmeans_cluster.fit(im_shaped)
    im_quant = kmeans_fitted.cluster_centers_[kmeans_fitted.labels_]
    im_ready = im_quant.reshape((shape[0], shape[1], 3))
    return im_ready
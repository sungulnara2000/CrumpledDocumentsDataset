import numpy as np
import glob
import os
import pickle
import cv2 as cv
from tqdm import tqdm
from PIL import Image
import pyheif
import pdf2image

from src import config


def get_file_name(file_path):
    return os.path.basename(file_path).split('.')[0]


def rint(x):
    ret = np.rint(x).astype(int)
    if type(x) == tuple:
        ret = tuple(ret)
    return ret


def nearest_odd(x):
    return int(np.ceil(x) // 2 * 2 + 1)


def ceil_even(number):
    ceil_even = int(np.ceil(number))
    if ceil_even % 2 != 0:
        ceil_even += 1
    return ceil_even


def reduce_in_size(image, n):
    return cv.resize(image, (rint(image.shape[1] / n), rint(image.shape[0] / n)))


def heic2png(image_path, delete=False):
    new_name = image_path.replace('HEIC', 'png')
    heif_file = pyheif.read(image_path)
    data = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    data.save(new_name, "PNG")
    if delete:
        os.remove(image_path)


def convert_heic_to_png(folder_path, delete=False):
    heics = glob.glob(f"{folder_path}/*.HEIC")
    for l in tqdm(heics):
        heic2png(l, delete)


def convert_single_pdf_to_image(pdf_path,
                                output_folder,
                                dpi=400,
                                fmt='png',
                                kwargs={}):
                                
    basename = os.path.basename(pdf_path).split('.')[0]
    pdf2image.convert_from_path(pdf_path,
                                dpi=dpi,
                                output_folder=output_folder,
                                fmt=fmt,
                                output_file=f'{basename}_dpi_{dpi}_',
                                **kwargs)


def convert_pdf_to_image(pdfs_folder, output_folder, dpi=400, fmt='png'):
    pdfs = glob.glob(f"{pdfs_folder}/*.pdf")
    for filename in tqdm(pdfs):
        kwargs = {}
        if 'british-airways' in filename:
            kwargs['first_page'] = 5
            kwargs['last_page'] = 13

        convert_single_pdf_to_image(filename,
                                    output_folder=output_folder,
                                    dpi=dpi,
                                    fmt=fmt,
                                    kwargs=kwargs)


def read_crumpled_image(file_path):
    img = cv.imread(file_path)
    if img.shape[0] < img.shape[1]:
        img = cv.rotate(img, cv.cv2.ROTATE_90_CLOCKWISE)
        cv.imwrite(file_path, img)

    return img


def get_rotation_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def rotate(angle: float,
           pivot: np.ndarray,
           points: np.ndarray):
    """
    angle: rotation angle in radians
    pivot: pivot point
    points: array with shape (n, 2) points to rotate around pivot
    """
    rotation_matrix = get_rotation_matrix(angle)

    pivot = pivot.reshape((1, 2))
    rotated = points - pivot
    rotated = np.dot(rotated, rotation_matrix.T)
    rotated += pivot

    return rotated


def get_line_pixels(P1, P2):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 2), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(int) + P1Y

    return itbuffer.astype(int)


def save_key_points(file_path, key_points):
    keys_points_as_tuple = [(point.pt, point.size) for point in key_points]
    pickle.dump(keys_points_as_tuple, open(file_path, "wb"))


def load_key_points(file_path):
    keys_points_as_tuple = pickle.load(open(file_path, "rb"))
    return [cv.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1]) for point in keys_points_as_tuple]



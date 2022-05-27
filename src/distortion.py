import numpy as np
import cv2 as cv
import os
from scipy import interpolate

from src import config, utils, detection, ordering, inpainting


class Distortion:
    def __init__(self,
                 crumpled_grid_path,
                 input_size,
                 output_size=(3024, 4032),
                 gray_illumination=True):
        """
        :param crumpled_grid_path: path to crumpled dot paper image
        :param input_size: tuple, (width, height) of input electronic documents
        :param output_size: tuple, (width, height) of result image
        :param gray_illumination: whether to use gray or colored illumination
        """
        self.paper_contour = None

        self.input_size = input_size
        self.output_size = output_size
        self.crumpled_grid_image = cv.imread(crumpled_grid_path)

        self.key_points = detection.detect_markers(crumpled_grid_path)
        self.ordered_key_points = ordering.order_markers(self.key_points)

        self.inpainted = inpainting.inpaint_key_points(crumpled_grid_path, self.key_points, gray=gray_illumination)
        if len(self.inpainted.shape) == 2:
          self.inpainted = cv.cvtColor(self.inpainted, cv.COLOR_GRAY2RGB)
          
        self.__interpolate_crumpling()

    def transform(self, document_path, background_path, resize=False):
        document_image = cv.imread(document_path)
        background_image = cv.imread(background_path)

        if resize:
            document_image = cv.resize(document_image, self.input_size)

        assert document_image.shape[1::-1] == self.input_size, \
            f'Size of document must be {self.input_size}, but document of size {document_image.shape[1::-1]} was given'

        transformed = self.__crumple(document_image)
        transformed = self.__add_illumination(transformed)
        transformed = self.__add_background(transformed, background_image)

        return transformed

    def __crumple(self, document_image):
        output_width, output_height = self.output_size

        transformed_document = np.full((output_height, output_width, 3), 0, dtype=np.uint8)
        transformed_document[self.interpolated[:, :, 1], self.interpolated[:, :, 0], :] = document_image

        return transformed_document

    def __interpolate_crumpling(self):
        photo_height, photo_width = self.crumpled_grid_image.shape[:2]
        document_width, document_height = self.input_size
        output_width, output_height = self.output_size

        crumpled_grid = self.ordered_key_points.copy()
        crumpled_grid[:, :, 1] *= output_height / photo_height
        crumpled_grid[:, :, 0] *= output_width / photo_width

        y_grid = np.linspace(0, document_height, config.N_ROWS)
        x_grid = np.linspace(0, document_width, config.N_COLS)

        self.document_pixels = np.transpose(np.indices((document_width, document_height)), (2, 1, 0))
        self.init_dot_positions = np.stack(np.meshgrid(x_grid, y_grid), axis=2).reshape((-1, 2))
        transformed_dot_positions = crumpled_grid.reshape((-1, 2))

        interpolated = interpolate.griddata(self.init_dot_positions,
                                            transformed_dot_positions,
                                            self.document_pixels,
                                            method='linear')
        interpolated = utils.rint(interpolated)

        assert np.all(interpolated >= 0)
        assert np.all(interpolated[:, :, 0] < output_width)
        assert np.all(interpolated[:, :, 1] < output_height)

        self.interpolated = interpolated

    def __find_paper_contour(self, transformed):
        if self.paper_contour is None:
          transformed_gray = cv.cvtColor(transformed, cv.COLOR_BGR2GRAY)
          # transformed_gray = cv.GaussianBlur(transformed_gray, (13, 13), 0)

          contours = cv.findContours(transformed_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
          contours = contours[0] if len(contours) == 2 else contours[1]

          big_contour = max(contours, key=cv.contourArea)

          self.paper_contour = np.zeros(self.crumpled_grid_image.shape)
          self.paper_contour = cv.drawContours(self.paper_contour, [big_contour], 0, (1, 1, 1), -1)

    def __add_illumination(self, transformed_document):
        self.__find_paper_contour(transformed_document)

        output_width, output_height = self.output_size

        illumination = self.inpainted * self.paper_contour
        illumination = cv.resize(illumination, (output_width, output_height))

        illumination_normalized = cv.normalize(illumination,
                                               None,
                                               alpha=0,
                                               beta=1,
                                               norm_type=cv.NORM_MINMAX,
                                               dtype=cv.CV_32F)
        transformed_document_w_illumination = illumination_normalized * transformed_document

        return transformed_document_w_illumination

    def __add_background(self, transformed_document, background_image):
        self.__find_paper_contour(transformed_document)

        output_width, output_height = self.output_size

        paper_contour = cv.resize(self.paper_contour, (output_width, output_height))
        background = cv.resize(background_image, (output_width, output_height))
        transformed_document = cv.resize(transformed_document, (output_width, output_height))

        transformed_document_w_background = background * np.logical_not(paper_contour) \
                                            + transformed_document * paper_contour

        return transformed_document_w_background

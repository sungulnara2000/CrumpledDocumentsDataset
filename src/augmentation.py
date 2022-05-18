import numpy as np
import cv2 as cv
from scipy import interpolate


from src import config, utils, detection
from src.distortion import Distortion


class Augmentation():
    def __init__(self,
                 distortion: Distortion,
                 decrease_rate_range=None,
                 angles_range=None,
                 width_margin=200,
                 height_margin=200):
        """
        :param distortion: Distortion object to sample augmentation from
        :param decrease_rate_range: np.array of numbers between 0 and 1, how much to reduce the size of the document
        :param angles_range: np.array of numbers between 0 and 180, how many degrees to rotate the document
        :param width_margin: distance in pixels from document to left and right edge of the output image
        :param height_margin: distance in pixels from document to top and bottom edge of the output image
        """
        self.distortion = distortion
        self._decrease_rate_range = decrease_rate_range
        self._angles_range = angles_range
        self._width_margin = width_margin
        self._height_margin = height_margin

        if decrease_rate_range is None:
            self._decrease_rate_range = np.arange(0.4, 0.6, 0.1)
        if angles_range is None:
            self._angles_range = np.radians(np.arange(0, 91))

        self.__sample_decrease_rate_and_angle()
        self.__make_initial_grid()
        self.__make_transformed_grid()

    def transform(self, document_path, background_path, resize=False):
        document_image = cv.imread(document_path)
        background_image = cv.imread(background_path)

        if resize:
            document_image = cv.resize(document_image, self.distortion.input_size)

        document_size = document_image.shape[1::-1]
        assert document_size == self.distortion.input_size, \
            f'Size of document must be {self.distortion.input_size}, but document of size {document_size} was given'

        crumpled = self.__crumple(document_image)
        crumpled_w_illumination = self.__add_illumination(crumpled)
        crumpled_w_background = self.__add_background(crumpled_w_illumination, background_image)

        return crumpled_w_background

    def __sample_decrease_rate_and_angle(self):
        """
        Randomly selects rectangle size and rotation angle
        so what rectangle fully fits into the document
        """
        document_width, document_height = self.distortion.input_size

        self.decrease_rate = np.random.choice(self._decrease_rate_range)

        self._cropped_rectangle_height = document_height * self.decrease_rate
        self._cropped_rectangle_width = document_width * self.decrease_rate

        angles_cos = np.cos(self._angles_range)
        angles_sin = np.sin(self._angles_range)

        bounding_box_widths = angles_cos * self._cropped_rectangle_width + angles_sin * self._cropped_rectangle_height
        bounding_box_heights = angles_cos * self._cropped_rectangle_height + angles_sin * self._cropped_rectangle_width

        acceptable_angles_ids = np.argwhere(
            (bounding_box_widths <= document_width) &
            (bounding_box_heights <= document_height)).ravel()

        acceptable_angles = self._angles_range[acceptable_angles_ids]
        angle_sign = np.random.choice([-1, 1])
        self.rotation_angle = np.random.choice(acceptable_angles) * angle_sign

    def __make_initial_grid(self):
        document_width, document_height = self.distortion.input_size

        initial_grid = self.distortion.init_dot_positions * self.decrease_rate
        initial_grid[:, 0] -= self._cropped_rectangle_width / 2
        initial_grid[:, 1] -= self._cropped_rectangle_height / 2

        initial_grid = utils.rotate(self.rotation_angle, np.array([0, 0]), initial_grid)

        rotated_w = utils.ceil_even(initial_grid[:, 0].max() - initial_grid[:, 0].min())
        rotated_h = utils.ceil_even(initial_grid[:, 1].max() - initial_grid[:, 1].min())

        x_center_range = np.arange(rotated_w / 2, document_width - rotated_w / 2, dtype=int)
        y_center_range = np.arange(rotated_h / 2, document_height - rotated_h / 2, dtype=int)

        assert len(x_center_range) > 0 and len(y_center_range) > 0, \
            'Bad parameters, rectangle does not fit the page'

        center_x = np.random.choice(x_center_range)
        center_y = np.random.choice(y_center_range)
        self.rectangle_center = np.array([center_x, center_y])

        initial_grid += self.rectangle_center
        self._initial_grid = utils.rint(initial_grid)

        assert np.all(0 <= initial_grid[0]) and np.all(initial_grid[0] < document_width) and \
               np.all(0 <= initial_grid[1]) and np.all(initial_grid[1] < document_height), f"""
        decrease_rate = {self.decrease_rate}, 
        rotation_angle = {self.rotation_angle}, 
        rectangle_center={self.rectangle_center}"""

    def __make_transformed_grid(self):
        # transform grid and it's center using whole crumpling function
        transformed_grid = self.distortion.interpolated[self._initial_grid[:, 1], self._initial_grid[:, 0], :]
        self._transformed_rectangle_center = self.distortion.interpolated[
            self.rectangle_center[1], self.rectangle_center[0]]

        # rotate grid back to the vertical position
        transformed_grid_rotated = utils.rotate(
            -self.rotation_angle,
            self._transformed_rectangle_center,
            transformed_grid)

        transformed_grid_rotated = utils.rint(transformed_grid_rotated)

        # find edge pixels of cropped rectangle
        initial_grid_corners = self._initial_grid[[0, -config.N_COLS, -1, config.N_COLS - 1], :]
        rectangle_edge_pixels = []

        for i in range(len(initial_grid_corners)):
            next_vertex = 0 if i + 1 == len(initial_grid_corners) else i + 1
            edge_pixels = utils.get_line_pixels(initial_grid_corners[i], initial_grid_corners[next_vertex])
            rectangle_edge_pixels.append(edge_pixels[:-1])

        rectangle_edge_pixels = np.concatenate(rectangle_edge_pixels)

        # find where edges of rectangle are transformed when crumpled
        self._transformed_rectangle_borders = self.distortion.interpolated[
                                             rectangle_edge_pixels[:, 1], 
                                             rectangle_edge_pixels[:, 0], :]

        transformed_rectangle_borders_rotated = utils.rotate(
            -self.rotation_angle,
            self._transformed_rectangle_center,
            self._transformed_rectangle_borders)

        transformed_rectangle_borders_rotated = utils.rint(transformed_rectangle_borders_rotated)

        # find bounding rectangle for the transformed grid
        x, y, w, h = cv.boundingRect(transformed_rectangle_borders_rotated)

        # adjust bounding rectangle so that it has required aspect ratio
        output_width, output_height = self.distortion.output_size
        output_aspect_ratio = output_height / output_width

        if h / w > output_aspect_ratio:
            change = int(np.ceil((h / output_aspect_ratio - w) / 2))
            w += 2 * change
            x -= change
        else:
            change = int(np.ceil((w * output_aspect_ratio - h) / 2))
            h += 2 * change
            y -= change

        assert np.abs(output_aspect_ratio - h / w) < 1e-2
        self._bounding_rectangle = {'center': (x, y), 'width': w, 'height': h}

        # expand grid to output size
        zero_coord = np.array([x, y], dtype=int)

        transformed_grid_aligned = (transformed_grid_rotated - zero_coord).astype(float)

        transformed_grid_aligned[:, 0] *= (output_width - 2 * self._width_margin) / w
        transformed_grid_aligned[:, 1] *= (output_height - 2 * self._height_margin) / h

        transformed_grid_aligned = utils.rint(transformed_grid_aligned)

        transformed_grid_aligned[:, 0] += self._width_margin
        transformed_grid_aligned[:, 1] += self._height_margin

        self._transformed_grid_aligned = transformed_grid_aligned

    def __crumple(self, document):

        assert document.shape[1::-1] == self.distortion.input_size, \
            f'Size of document must be {self.distortion.input_size}, but document of size {document.shape[1::-1]} was given'

        output_width, output_height = self.distortion.output_size

        interpolated = interpolate.griddata(
            self.distortion.init_dot_positions,
            self._transformed_grid_aligned,
            self.distortion.document_pixels,
            method='linear')
        interpolated = utils.rint(interpolated)

        assert np.all(interpolated >= 0)
        assert np.all(interpolated[:, :, 0] < output_width)
        assert np.all(interpolated[:, :, 1] < output_height)

        transformed_document = np.full((output_height, output_width, 3), 255, dtype=np.uint8)
        transformed_document[interpolated[:, :, 1], interpolated[:, :, 0], :] = document

        return transformed_document

    def __add_illumination(self, transformed_document):
        output_width, output_height = self.distortion.output_size

        paper_contour = cv.drawContours(np.zeros(self.distortion.crumpled_grid_image.shape),
                                        [self._transformed_rectangle_borders],
                                        0, (1, 1, 1), -1)

        inpainted_part = self.distortion.inpainted * paper_contour

        rotation_matrix = cv.getRotationMatrix2D(
            tuple(self._transformed_rectangle_center), 
            np.degrees(self.rotation_angle), 
            1.0)
        
        inpainted_part = cv.warpAffine(
            inpainted_part, 
            rotation_matrix, 
            inpainted_part.shape[1::-1], 
            flags=cv.INTER_LINEAR)

        x, y = self._bounding_rectangle['center']
        illumination = cv.resize(
            inpainted_part[y: y + self._bounding_rectangle['height'], 
                           x: x + self._bounding_rectangle['width'], :],
            (output_width - 2 * self._width_margin, output_height - 2 * self._height_margin))

        illumination_w_margin = np.zeros((output_height, output_width, 3))
        illumination_w_margin[self._height_margin: -self._height_margin,
                              self._width_margin: -self._width_margin, :] = illumination

        illumination_normalized = cv.normalize(illumination_w_margin, 
                                               None, 
                                               alpha=0, beta=1, 
                                               norm_type=cv.NORM_MINMAX, 
                                               dtype=cv.CV_32F)
        transformed_document_w_illumination = illumination_normalized * transformed_document

        return transformed_document_w_illumination

    def __add_background(self, transformed_document, background_image):
        output_width, output_height = self.distortion.output_size

        paper_contour, _ = detection.find_biggest_contour(transformed_document)
        paper_contour_image = np.ones(transformed_document.shape)
        paper_contour_image = cv.drawContours(paper_contour_image, [paper_contour], 0, (0, 0, 0), -1)

        background = cv.resize(background_image, (output_width, output_height))

        transformed_document_w_background = background * paper_contour_image + transformed_document
        return transformed_document_w_background

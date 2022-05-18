import numpy as np
import cv2 as cv
import alphashape
from google.colab.patches import cv2_imshow


from src import config
from src import utils


def get_corners_of_contour(contour, epsilon=0.04):
  """
  Return corners of rectangle-like contour.
  The order is: top-left, bottom-left, bottom-right, top-right
  """

  perimeter = cv.arcLength(contour, closed=True)
  corners = cv.approxPolyDP(contour, epsilon * perimeter, closed=True).squeeze()

  top_corner_index = np.argmin(corners[:, 1])
  corners = np.roll(corners, -top_corner_index, axis=0)

  if np.linalg.norm(corners[0] - corners[1]) < np.linalg.norm(corners[2] - corners[1]):
    corners = np.roll(corners, -1, axis=0)

  assert len(corners) == 4, f"Number of corners is {len(corners)} != 4"

  return corners


def get_frame_indices(top_left, h, w):
  """
  returns counter-clockwise frame indices starting from top-left index
  """
  x, y = top_left
  x_indices = np.concatenate((np.repeat(x, h - 1), 
                              np.arange(x, x + w - 1), 
                              np.repeat(x + w - 1, h - 1), 
                              np.arange(x + w - 1, x, -1)))
  

  y_indices = np.concatenate((np.arange(y, y + h - 1), 
                              np.repeat(y + h - 1, w - 1), 
                              np.arange(y + h - 1, y, -1), 
                              np.repeat(y, w - 1)))
  
  return x_indices, y_indices


def order_markers(key_points, alpha=0.005):
  ordered_kp = np.full((config.N_ROWS, config.N_COLS, 2), fill_value=-1, dtype=float)
  unvisited_points = set([x.pt for x in key_points])

  frame_height = config.N_ROWS
  frame_width = config.N_COLS

  i = 0
  while frame_height > 1 and frame_width > 1:
    points = list(unvisited_points)

    hull = alphashape.alphashape(points, alpha)
    hull_points = hull.exterior.coords.xy
    hull_points = np.stack(hull_points, axis=1)[:-1]

    corners = get_corners_of_contour(utils.rint(hull_points))

    top_left_index = np.argmin(np.linalg.norm(hull_points - corners[0], axis=1))
    hull_points = np.roll(hull_points, -top_left_index, axis=0)


    frame_indices_x, frame_indices_y = get_frame_indices([i, i], frame_height, frame_width)
    ordered_kp[frame_indices_y, frame_indices_x] = hull_points

    unvisited_points.difference_update(list(map(tuple, hull_points)))
    frame_height -= 2
    frame_width -= 2
    i += 1
  

  center_points = np.array(sorted(
    list(map(np.array, list(unvisited_points))),
    key=lambda x: x[1])
    )
  ordered_kp[i: i + frame_height, i] = center_points


  return ordered_kp


def plot_ordered(img, ordered_kp, show_size=(756, 1008)):
  img_ordered_markers = img.copy()

  for row in range(config.N_ROWS):
    for col in range(config.N_COLS):
        cv.putText(
          img_ordered_markers, 
          f'{col}',
          tuple(utils.rint(ordered_kp[row][col])), 
          cv.FONT_HERSHEY_SIMPLEX, 
          1,
          (255, 0, 0), 
          2,
          cv.LINE_AA)
        
        if col < config.N_COLS - 1:
          img_ordered_markers = cv.arrowedLine(img_ordered_markers, 
                              tuple(utils.rint(ordered_kp[row][col])), 
                              tuple(utils.rint(ordered_kp[row][col + 1])),
                              (0, 0, 0), 2)
        if row < config.N_ROWS - 1:
          img_ordered_markers = cv.arrowedLine(img_ordered_markers, 
                              tuple(utils.rint(ordered_kp[row][col])), 
                              tuple(utils.rint(ordered_kp[row + 1][col])),
                              (0, 0, 0), 2)
        
  cv2_imshow(cv.resize(img_ordered_markers, show_size))
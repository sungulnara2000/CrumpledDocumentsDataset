import numpy as np
import os
import cv2 as cv
from skimage import feature

from src import config
from src import utils


def intersection_area(d, R, r):
    """
    Return the area of intersection of two circles.
    The circles have radii R and r, and their centres are separated by d.
    """

    if d <= abs(R-r):
        # One circle is entirely enclosed in the other.
        return np.pi * min(R, r)**2
    if d >= r + R:
        # The circles don't overlap at all.
        return 0

    r2, R2, d2 = r**2, R**2, d**2
    alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
    beta = np.arccos((d2 + R2 - r2) / (2*d*R))
    return ( r2 * alpha + R2 * beta -
             0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta))
           )


def iou_key_points(kp_one, kp_two):
  distance_between_centers = np.linalg.norm(np.array(kp_one.pt) - np.array(kp_two.pt))
  radius_one, radius_two = kp_one.size / 2, kp_two.size / 2
  intersection = intersection_area(distance_between_centers, radius_one, radius_two)
  area_one = np.pi * radius_one ** 2
  area_two = np.pi * radius_two ** 2
  return intersection / (area_one + area_two - intersection)


def delete_similar_key_points(kp, iou_threshold):
  """
  Non maximum suppression for keypoints

  Note that must be in descending response order as from DetectAndCompute
  """

  take_mask = np.full((len(kp)), True, dtype=bool)

  for i, point in enumerate(kp):
    if take_mask[i] == 0:
      continue

    for compare_point_idx in range(i + 1, len(kp)):
      if iou_key_points(point, kp[compare_point_idx]) >= iou_threshold:
        take_mask[compare_point_idx] = False

  new_kp = np.array(kp)[take_mask]
  indices = np.arange(len(kp))[take_mask]

  return new_kp, indices



def find_biggest_contour(image):
  """
  returns: biggest contour and all contours
  """
  img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  img_blur = cv.GaussianBlur(img_gray, (13, 13), 0)

  edges = feature.canny(img_blur, sigma=0.5)
  edges = edges.astype('uint8') * 255

  kernel = np.ones((7,7), np.uint8)
  closing = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
  closing = cv.GaussianBlur(closing, (7, 7), 0)


  contours = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]

  big_contour = max(contours, key=cv.contourArea)

  # make biggest contour slightly smaller
  img_contour = np.zeros(image.shape[:2], dtype='uint8')
  cv.drawContours(img_contour, [big_contour], -1, (255), -1)
  cv.drawContours(img_contour, [big_contour], -1, (0), 10)

  contours_helper = cv.findContours(img_contour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  contours_helper = contours_helper[0] if len(contours_helper) == 2 else contours_helper[1]

  big_contour = max(contours_helper, key=cv.contourArea)

  return big_contour, contours


def make_template_circle(size=32):
  reference_circle = np.full((size, size), 255, np.uint8)
  reference_circle = cv.circle(reference_circle, 
                              utils.rint((size/2, size/2)), 
                              utils.rint(size/5), 
                              color=(0, 0, 0), 
                              thickness=-1)
  
  surf = cv.xfeatures2d.SURF_create(hessianThreshold=700, nOctaves=2, nOctaveLayers=2, upright=False)
  reference_circle_kp, reference_circle_des = surf.detectAndCompute(reference_circle, None)

  return reference_circle_des
  

def detect_markers(img_path):
  basename = utils.get_file_name(img_path)
  save_path = os.path.join(config.KEY_POINTS_FOLDER, basename + '.pickle')
  
  if os.path.exists(save_path):
    circle_key_points = utils.load_key_points(save_path)
  else:
    img = cv.imread(img_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (15, 15), 0)

    surf = cv.xfeatures2d.SURF_create(hessianThreshold=1500, upright=False)
    key_points, descriptors = surf.detectAndCompute(img_blur, None)

    strongest_key_points, indices = delete_similar_key_points(key_points, iou_threshold=0.2)
    strongest_descriptors = descriptors[indices]

    reference_circle_des = make_template_circle()
    circle_kp_des = sorted(list(zip(strongest_key_points, strongest_descriptors)), 
                          key=lambda x: np.linalg.norm(x[1] - reference_circle_des))
    circle_key_points = [x[0] for x in circle_kp_des][:config.N_COLS * config.N_ROWS]

    utils.save_key_points(save_path, circle_key_points)

  return circle_key_points
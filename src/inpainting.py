import numpy as np
import os
import cv2 as cv

from src import config, utils


def inpaint_key_points(img_path, key_points, gray=True):
  basename = utils.get_file_name(img_path)
  img = cv.imread(img_path)

  if gray:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

  mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
  for kp in key_points:
    mask = cv.circle(mask, utils.rint(kp.pt), utils.rint(kp.size / 2), (255), -1)

  save_name = f'inpainted_{"gray_" if gray else ""}{basename}.png'
  save_path = os.path.join(config.INPAINTED_FOLDER, save_name)

  if os.path.exists(save_path):
    inpainted = cv.imread(save_path)
  else:
    inpainted = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)
    cv.imwrite(save_path, inpainted)


  return inpainted

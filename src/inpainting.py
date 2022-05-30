import numpy as np
import os
import cv2 as cv

from src import config, utils


def inpaint_key_points(img_path, key_points, gray=True, overwrite=False):
  basename = utils.get_file_name(img_path)
  img = cv.imread(img_path)

  if gray:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

  mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
  for kp in key_points:
    mask = cv.circle(mask, utils.rint(kp.pt), utils.rint(kp.size / 2), (255), -1)

  mask = cv.bitwise_not(mask)

  save_name = f'inpainted_fsrfast_{"gray_" if gray else ""}{basename}.png'
  save_path = os.path.join(config.INPAINTED_FOLDER, save_name)

  if not overwrite and os.path.exists(save_path):
    inpainted = cv.imread(save_path)
  else:
    inpainted = img.copy()
    cv.xphoto.inpaint(img, mask, inpainted, cv.xphoto.INPAINT_FSR_FAST)
    cv.imwrite(save_path, inpainted)


  return inpainted

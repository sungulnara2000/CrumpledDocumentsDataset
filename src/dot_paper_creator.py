import cv2
import numpy as np

def make_dot_paper(height=3508, width=2480, colored=True):
  img = np.full((height, width, 3), 255, np.uint8)

  radius = 15
  step_h = 118
  pad_h = int(np.ceil((height % step_h) / 2))
  if pad_h < 2*radius:
      pad_h = 2*radius

  step_w = 120
  pad_w = int(np.ceil((width % step_w) / 2))
  if pad_w < 2*radius:
      pad_w = 2*radius
  
  width_ids = range(pad_w, width, step_w)
  height_ids = range(pad_h, height, step_h)

  w_color_range = np.linspace(0, 255, len(width_ids))
  h_color_range = np.linspace(0, 255, len(height_ids))

  for i, w in enumerate(width_ids):
    for j, h in enumerate(height_ids):
      color = (h_color_range[j], 0, w_color_range[i]) if colored else (0, 0, 0)
      cv.circle(img, (w, h), radius, color, -1)

  return img
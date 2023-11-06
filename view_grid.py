import numpy as np
import imageio
import cv2 as cv

cut_out_folder = 'cut_out_folder'

im = imageio.imread('zebrafish.png')
im = imageio.imread('well_plates_bgsub.png')
im = np.asarray(im)

grid = np.load('grid.npy')
y = 0
x = 0
for idx, circ in enumerate(grid):
    center_x , center_y, radius = circ.astype(int)
    top_left = (center_y - radius, center_x - radius)
    bottom_right = (center_y + radius, center_x + radius)
    cut_out = im[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]
    cY, cX = cut_out.shape[:2]
    if cY > y: y = cY
    if cX > x: x = cX

    cv.imwrite(cut_out_folder + '/cutout' + str(idx) + '.png', cut_out)
print('biggest y: ', y)
print('biggest x: ', x)





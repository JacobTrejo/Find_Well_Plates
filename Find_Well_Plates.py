import numpy as np
import cv2 as cv
import imageio

def pop_element_of_2D_array(arr, idx):
    h, w = arr.shape
    newArr = np.zeros((h - 1, w))

    newArr[:idx] = arr[:idx]
    newArr[idx:] = arr[idx + 1:]

    poped_element = arr[idx,...]

    return newArr, poped_element

def get_circles_of_similar_radius(circ_arr):

    def iterator(circ_arr, grouped_list):
        threshold = 5
        circ = circ_arr[0,...]
        radius = circ[2]

        grouped_array = circ
        # First row is temporary
        ungrouped_array = np.zeros(circ.shape)
        for idx, other_circ in enumerate(circ_arr[1:,...]):
            other_radius = other_circ[2]
            radius_diff = np.abs(other_radius - radius)
            if radius_diff < threshold:
                # Got to do this because numpy is wierd
                if grouped_array.ndim == 2:
                    grouped_array = np.concatenate((grouped_array, other_circ[np.newaxis]), axis=0)
                else:
                    grouped_array = np.stack((grouped_array, other_circ), axis=0)
            else:
                # Got to do this because numpy is wierd
                if ungrouped_array.ndim == 2:
                    ungrouped_array = np.concatenate((ungrouped_array, other_circ[np.newaxis]), axis=0)
                else:
                    ungrouped_array = np.stack((ungrouped_array, other_circ), axis=0)
                # ungrouped_array = np.concatenate((ungrouped_array, other_circ), axis = 0)
        grouped_list.append(grouped_array)

        circles_left = len(ungrouped_array) - 1
        if ungrouped_array.ndim == 1: circles_left = 0
        if circles_left >= 2:
            circ_arr = ungrouped_array[1:,:]
            iterator(circ_arr, grouped_list)

        return grouped_list

    grouped_list = iterator(circ_arr, [])

    # finding the group with the most amount of circles
    largest_amount = 0
    idx_of_largest = 0
    for idx, grouped_arr in enumerate(grouped_list):
        amount = len(grouped_arr)
        if largest_amount <= amount:
            largest_amount = amount
            idx_of_largest = idx

    return grouped_list[idx_of_largest]

def find_top_left_circ_and_bottom_right(circ_arr):
    x_min = np.min(circ_arr[:,0])
    y_min = np.min(circ_arr[:,1])

    x_max = np.max(circ_arr[:,0])
    y_max = np.max(circ_arr[:,1])

    average_radius = np.mean(circ_arr[:,2])
    return np.array([[x_min, y_min, average_radius],
                     [x_max, y_max, average_radius]])

def create_circ_grid(top_left_circ, bottom_right, rows, columns):
    sx, sy = top_left_circ[0:2]
    bx, by = bottom_right[0:2]

    width = (bx - sx) / (columns - 1)
    height = (by - sy) / (rows - 1)

    # Divide width + height by 2 to get average, divide by 2 again to get radius
    radius = (width + height) / 4


    grid_of_circles = np.zeros((rows * columns, 3))
    for row in range(rows):
        for column in range(columns):
            center_x = sx + column * width
            center_y = sy + row * height

            idx = column + (row * columns)
            grid_of_circles[idx, :] = [center_x, center_y, radius]

    return grid_of_circles
# temp = np.array([1,1,1])
# temp2 = np.array([2,2,2])
# temp2D = np.array([[3,3,3],
#                    [4,4,4]])
#
# base = temp2D
# add = temp2
# if base.ndim == 2:
#     result = np.concatenate((base, add[np.newaxis]), axis = 0)
# else:
#     result = np.stack((base, add), axis = 0)

def error_between_actual_and_grid(circ_arr, grid_of_circles):
    # NOTE: it might be a good idea to take out circles in the grid that have been paired with a circ
    # this is in case some circles start to get paired with similar points on the grid

    average_radius = circ_arr[0,2]

    threshold_percent_to_be_a_good_circ = .2
    distance_threshold = average_radius * threshold_percent_to_be_a_good_circ

    amount_of_good_circs = 0
    minimum_distances = []
    for circ in circ_arr:
        differance_arr = grid_of_circles - circ
        distance_arr = (differance_arr[:,0]** 2 + differance_arr[:,1]**2) ** .5
        min_distance = np.min(distance_arr)

        if min_distance < distance_threshold:
            amount_of_good_circs += 1

        minimum_distances.append(min_distance)
    minimum_distances = np.array(minimum_distances)

    average_min_dist = np.mean(minimum_distances)

    return amount_of_good_circs, average_min_dist

def error(circles, rows, columns):
    circles = get_circles_of_similar_radius(circles)
    top_and_bottom = find_top_left_circ_and_bottom_right(circles)
    grid_of_circles = create_circ_grid(top_and_bottom[0], top_and_bottom[1], rows, columns)
    amount_of_good_circles, average_distance = error_between_actual_and_grid(circles, grid_of_circles)
    return amount_of_good_circles, average_distance, grid_of_circles

def find_circles(im_path):

    src = cv.imread(im_path, cv.IMREAD_COLOR)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    gray = cv.medianBlur(gray, 5)

    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=100, param2=30,
                              minRadius=1, maxRadius=70)

    if circles is not None:
        circles = circles[0,:]
    should_group = True
    if should_group:
        if circles is not None:
            # circles = get_circles_of_similar_radius(circles)
            # top_and_bottom = find_top_left_circ_and_bottom_right(circles)
            # grid_of_circles = create_circ_grid(top_and_bottom[0],top_and_bottom[1] , 6, 8)
            # amount_of_good_circles, average_distance = error(circles, grid_of_circles)
            amount_of_good_circles, average_distance = error(circles, 6, 8)
            jj = 5

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for idx, i in enumerate(circles):
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
    if should_group:
        cv.imwrite('result_with_grouping.png', src)
    else:
        cv.imwrite('result.png', src)

    return 0

def find_best_set_of_circles(im_path, rows_for_well_plate, columns):
    src = cv.imread(im_path, cv.IMREAD_COLOR)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    gray = cv.medianBlur(gray, 5)

    rows = gray.shape[0]

    total_amount_of_circles = rows_for_well_plate * columns
    amount_of_good_circles_so_far = 0
    best_average_distance_so_far = np.inf
    current_best_error = (amount_of_good_circles_so_far, best_average_distance_so_far)
    current_best_grid = None

    # Calculation the values we will try
    smallest_dimension = np.min(gray.shape[:2])
    smallest_possible_radius = smallest_dimension * .05
    biggest_possible_radius = smallest_dimension * .15
    max_resolution = 50
    if (biggest_possible_radius - smallest_possible_radius) + 1 < max_resolution:
        max_resolution = int( (biggest_possible_radius - smallest_possible_radius) + 1)

    possible_radius = np.linspace(smallest_possible_radius, biggest_possible_radius, max_resolution)
    possible_radius = possible_radius.astype(int)

    # Finding the best grid
    for big_radius in possible_radius:
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                                  param1=100, param2=30,
                                  minRadius=1, maxRadius=big_radius)

        if circles is not None:
            circles = circles[0, :]
        else:
            continue
        amount_of_circles = len(circles)
        if amount_of_circles/ total_amount_of_circles < .1:
            # The amount of circles is too few to give a good estimate
            continue
        else:
            amount_of_good_circles, average_distance, grid_of_circles = error(circles, rows_for_well_plate, columns)
            if amount_of_good_circles >= amount_of_good_circles_so_far:
                if average_distance < best_average_distance_so_far:
                    amount_of_good_circles_so_far = amount_of_good_circles
                    best_average_distance_so_far = average_distance
                    current_best_grid = grid_of_circles

    if current_best_grid is not None:
        np.save('grid.npy', current_best_grid)

        # Drawing
        circles = current_best_grid
        circles = np.uint16(np.around(circles))
        for idx, i in enumerate(circles):
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
        cv.imwrite('result.png', src)









im_path = 'zebrafish.png'
# find_circles(im_path)

# im_path = 'other_well_plate_image.jpg'
# im_path = 'well_plate4.png'

find_best_set_of_circles(im_path, 6,8)





















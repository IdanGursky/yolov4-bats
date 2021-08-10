import numpy as np
from core.object_utils import intersection_larger_then_limit
import core.constants as const


# frame number? holding a ttl list in an additonal dict?
# event_dict = {hole0: (frame_number, times), etc'
# holding curr and last arrays as well? [


# when do we initialize? 
# perhaps when env_established returns true (just once)
def event_matrix_initializer():
    # init a 20 by 14 event matrix, 20 frames and 14 elements to check overlap for
    # np.zeroes(shape = (# rows, # cols))
    # dtype bool should take up the minimum amount of space
    event_matrix = np.empty((1,14),dtype=bool)
    
    # hold the matrix in detect_video_adjusted and send it each time
    return event_matrix 
    

def hole_overlap(holes_list, bat_coor, ratio_limit=const.hole_overlap_ratio_limit):
    binary_holes_list = [0] * len(holes_list)
    binary_holes_list = [1 if intersection_larger_then_limit(coor, bat_coor, ratio_limit) else 0 for coor in holes_list]
    
    return np.array(binary_holes_list)

# lower box threshold to 5%
def box_overlap(box_list, bat_coor, ratio_limit=const.box_overlap_ratio_limit):
    binary_box_list = [0] * len(box_list)
    binary_box_list = [1 if intersection_larger_then_limit(coor, bat_coor, ratio_limit) else 0 for coor in box_list]
    
    return np.array(binary_box_list)


# builds a row in event matrix - [7 holes followed by 6 boxes]
# row = [hole_0, hole_1, hole_2, hole_3, hole_4, hole_5, carrier, box_0, box_1, box_2, box_3,  box_4, box_5]
def events_row_builder(box_list, holes_list, bat_coor):
    holes_array = hole_overlap(holes_list, bat_coor)
    box_array = box_overlap(box_list, bat_coor)
    
    # create a 13 bits array of 0 and 1, based on overlapping in this frame
    frame_array = np.concatenate([holes_array,box_array])

    return frame_array
    
# default not found, so coordinates are unnecessary 
def event_matrix_updater(event_matrix_file, event_matrix, box_dict, holes_dict, bat_coor=0, found=0):
    if found:
        frame_array = events_row_builder(box_dict, holes_dict, bat_coor)
        # add the last col that indicates if the bat is in frame or not 
        frame_array = np.append(frame_array, [1])
    else:
        frame_array = np.zeros(14)
    
    # first row is made up due to initializer
    event_matrix = np.append(event_matrix, [frame_array], axis=0)
    # write each row to a csv file
    np.savetxt(event_matrix_file, frame_array.reshape(1, frame_array.shape[0]), delimiter=',')
    
    return event_matrix
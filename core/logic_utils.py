import numpy as np
import csv
import core.logger_utils as logger_utils
import core.constants as const
 
def remove_small_intervals(streaks_list, threshold):

    small_streak_marker = [0] * len(streaks_list)
    
    print("streaks_list before removal: ", streaks_list)

    for i in range(len(streaks_list)):
        streak = streaks_list[i]
        streak_length = streak[1] - streak[0]
        print("streak: ", streak)
        print("streak_length: ", streak_length)
        if streak_length < threshold:
            small_streak_marker[i] = 1
    
    for i in range(len(small_streak_marker))[::-1]:
        if small_streak_marker[i] == 1:
            print("small streak detected in index: ", i)
            streaks_list.pop(i)
     
     
    print("streaks_list after removal: ", streaks_list)
 
    return streaks_list

def recursive_merge(inter, threshold, start_index = 0):
    for i in range(start_index, len(inter) - 1):
        if inter[i][1] + threshold > inter[i+1][0]:
            new_start = inter[i][0]
            new_end = inter[i+1][1]
            inter[i] = [new_start, new_end]
            del inter[i+1]
            return recursive_merge(inter.copy(), threshold, start_index=i)
    return inter 
    
    
def merge_and_clean_array(streaks_list_array, small_intervals_threshold, distance_threshold, length_threshold): 
    
    small_intervals_filtered_array = []
    merged_streaks_array = []
    cleaned_streaks_array = []
    
    # filter very small intervals, smaller than small_intervals_threshold
    for streaks_list in streaks_list_array:
        if streaks_list:
            new_streak_list = remove_small_intervals(streaks_list, small_intervals_threshold)
            small_intervals_filtered_array.append(new_streak_list)
        else:
            small_intervals_filtered_array.append([])
    
    # merge intervals that have a minimum distance_threshold between them
    # exclude the found? col out of the merge 
    for i in range(len(small_intervals_filtered_array)):
        streaks_list = small_intervals_filtered_array[i]
        if i < 13:
        # merge holes into boxes?
            if i > 6:
                holes_streaks_list = small_intervals_filtered_array[i-7]
                streaks_list = sorted(streaks_list + holes_streaks_list)
            new_streak_list = recursive_merge(streaks_list, distance_threshold)
        else:
            new_streak_list = streaks_list
        merged_streaks_array.append(new_streak_list)
        
    # refiltering slightly larger intervals
    # for streaks_list in merged_streaks_array:
        # if streaks_list:
            # new_streak_list = remove_small_intervals(streaks_list, length_threshold)
            # cleaned_streaks_array.append(new_streak_list)
        # else:
            # cleaned_streaks_array.append([])
         

    # return cleaned_streaks_array
    
    return merged_streaks_array
    

# the idea is to fetch the start and end of the streaks of 1's in each col
# output is a list of arrays, each element is the array of streaks for each col -   [array(streak_start_index, streak_end_index),...]
def streak_indices(event_matrix):
    streaks_array = []
    
    for col in event_matrix.T:
        # get slices of clumps of 1's in a column
        slices = np.ma.clump_masked(np.ma.masked_where(col, col))
        # convert each slice to a list of indices.
        result = [[s.start, s.stop - 1] for s in slices]
        streaks_array.append(result)
    
    return streaks_array


def intersect(first_streak,second_streak):
    intersection = min(first_streak[1], second_streak[1]) - max(first_streak[0], second_streak[0])
    if intersection > 0:
            return True
    
    return False
        
# return true / false if the box and hole has overlapping timeframe
# if they overlap 
def box_hole_timeframe_overlap(hole_streaks_list, box_streak, box_index):
    # box and hole are always 7 indices apart
    overlap = False
    new_streak = box_streak
    hole_streak = 0
    
    for hole_streak in hole_streaks_list:
        if intersect(box_streak,hole_streak):
            overlap = True 
            new_start = min(hole_streak[0], box_streak[0])
            new_end = max(hole_streak[1], box_streak[1])
            new_streak = (new_start, new_end)
            break
    
    return overlap, new_streak, hole_streak
    
    
def entered_the_box(found_streak_list, box_overlap_end, entered_frame_count, next_streak):
    
    print("checking if entered the box")
    range_to_check = [box_overlap_end + entered_frame_count/10, box_overlap_end+entered_frame_count]
    print("range_to_check: " ,range_to_check)
    for streak in found_streak_list:
        # if there's an intersection - the bat didn't entered the box
        if intersect(range_to_check,streak):
            return (False, 0, 0)
    
    in_row = box_overlap_end + 1
    
    # needs to be tightened up
    out_row = next_streak[0]
    
    return (True, in_row, out_row)
    
def remove_overlapping_intervals(found_col_set, events_intervals_set):

    overlap_marker = [0] * len(found_col_set)
    
    for i in range(len(found_col_set)):
        found_streak = found_col_set[i]
        for event_streak in events_intervals_set:
            if intersect(event_streak,found_streak):
                overlap_marker[i] = 1
    
    for i in range(len(overlap_marker))[::-1]:
        if overlap_marker[i] == 1:
            print("exploring overlap detected in index: ", i)
            found_col_set.pop(i)
            
    return found_col_set

def event_sorter(f, streaks_list_array, oa_fps, length_threshold):

    '''
      Events - Landed, 
               Landed and looking in, 
               Landed and entered the box (with in and out times)
               Exploring
    '''          
    
    events_intervals_set = []
        
    for box_index in range(6,13):
        streak_list = streaks_list_array[box_index]
        if streak_list:
            for i in range(len(streak_list)):
                
                streak = streak_list[i]
                
                streak_len = streak[1] - streak[0]
                if streak_len < length_threshold:
                    continue
                    
                event = "Landed"
                in_row, out_row = 0,0
                start = streak[0]
                end = streak[1]
                # carrier
                if box_index == 6:
                    logger_utils.logger(f, start, end, oa_fps, box_index, event)
                    events_intervals_set.append(list(streak))
                # boxes - indices 7-12
                elif box_index > 6:
                    hole_index = box_index - 7
                    hole_streaks_list = streaks_list_array[hole_index]
                    overlap, new_streak, hole_streak = box_hole_timeframe_overlap(hole_streaks_list, streak,box_index)
                    if overlap:
                        start = new_streak[0]
                        end = hole_streak[1]
                        
                        # Entered the box
                        found_streak_list = streaks_list_array[13]
                        
                        # Make sure not to "fall over" the edge of the array
                        if i + 1 < len(streak_list):
                            next_streak = streak_list[i+1]
                            entered, in_row, out_row = entered_the_box(found_streak_list, end, const.entered_the_box_frame_count, next_streak)
                        else:
                            entered = False
                            
                        if entered:
                            # Skip the next streak, as it marks the outing of the bat
                            i += 1
                            print("entered the box")
                            event += " and Entered the box"
                        # Landing and looking in 
                        else:
                            event += " and Looking in"

                    logger_utils.logger(f, start, end, oa_fps, box_index, event, in_row, out_row)
                    events_intervals_set.append(list(new_streak))
    '''
    Exploring
        compare found_col intervals with the events joined by now
        drop every interval that overlap
        recursive_merge the left ones
    '''
    
    
    event = "Exploring"
    print("events_intervals_set: ", events_intervals_set)
    found_col_set = streaks_list_array[13]
    print("found_col_set: ", found_col_set)
    merged =  recursive_merge(found_col_set, const.minimum_distance_to_merge)
    print("merged: ", merged)
    
    exploring_set = remove_overlapping_intervals(found_col_set, events_intervals_set)
    exploring_set = recursive_merge(exploring_set, const.minimum_distance_to_merge)
    for streak in exploring_set:
        exp_index = 13
        in_row, out_row = 0, 0 
        logger_utils.logger(f, streak[0], streak[1], oa_fps, exp_index, event, in_row, out_row)
  
  
def events_logger(directory, logger, event_matrix, oa_fps, log_path, video_name):
        
    # remove first row of event_matrix 
    event_matrix = np.delete(event_matrix, 0, axis = 0)
    
    streaks_list_array = streak_indices(event_matrix)
    
    streaks_list_array = merge_and_clean_array(streaks_list_array,const.very_small_intervals_filtering_length, const.minimum_distance_to_merge, const.minimum_interval_length)
    
    for i in range(len(streaks_list_array) - 1):
       obj = logger_utils.index_to_object(i)
       print(obj, " : " , streaks_list_array[i])
    
    print("streaks_list_array: ", streaks_list_array)
    
    event_sorter(logger, streaks_list_array, oa_fps, const.minimum_interval_length)
    logger.close()
    logger_utils.sort_log_by_time(directory, log_path, video_name)

    return 
    

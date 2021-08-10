###### Holds all the constants ######


### constants that effect event matrix / data_csv creation -

# the amount that the bat has to overlap with a hole to count as '1' in event matrix 
# range - [0,1]
hole_overlap_ratio_limit = 0.3

# the amount that the bat has to overlap with a box to count as '1' in event matrix 
# range - [0,1]
box_overlap_ratio_limit = 0.1


### constants that effect logic applied on the event matrix -

# numbers of frames the bat isn't 'found' (14th col in event matrix) that is declared as 'entered the box'
entered_the_box_frame_count = 100

# preliminary filtering of very small intervals
very_small_intervals_filtering_length = 3 

# minimum numbers of frame between intervals to merge between them to one interval
minimum_distance_to_merge = 30
    
# minimum numbers of frames in an interval so he counts as en event
minimum_interval_length = 30 
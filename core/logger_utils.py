import numpy as np
import csv
import os

########### output files handling 

# deleting file in path, if exist
def delete_if_exist(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)
        print("deleted old file")
    return

# add top row for log csv
def add_headers(f):
    f.write(str("event") + ',' + str("object") + ',' + str("start time") + ',' + str("end time") + 
    ',' + str("in time") + ',' + str("out time") + '\n')
    return 
    
def file_init(path, binary):
    delete_if_exist(path)
    if binary:
        file = open(path, "ab")
    else:
        file = open(path, "a")
    return file
    
def xy_init(directory, xy_name, video_name):
    xy_path = directory + "/" + xy_name + "_" + video_name + ".csv"
    print("xy_path: ", xy_path)
    return file_init(xy_path, False)

def logger_init(directory, log_name, video_name):
    log_path = directory + "/" + log_name + "_" + video_name + ".csv"
    print("log_path: ", log_path)
    log_file = file_init(log_path, False)
    add_headers(log_file)
    return log_path, log_file
    
def event_matrix_init(directory, event_matrix_name, video_name):
    event_matrix_path = directory + "/" + event_matrix_name + "_" + video_name + ".csv"
    print("event_matrix_path: ", event_matrix_path)
    event_matrix_file = file_init(event_matrix_path, True)
    return event_matrix_file

def output_init(directory, event_matrix_name, xy_name, log_name, video_name):
    event_matrix_file = event_matrix_init(directory, event_matrix_name, video_name)
    xy_file = xy_init(directory, xy_name, video_name)
    log_path, log_file = logger_init(directory, log_name, video_name)
    
    return event_matrix_file, xy_file, log_path, log_file   
    

########### logger logic

# return (hours, minutes, seconds)
def time_calculator2(row_number, oa_fps):
    duration = row_number/oa_fps
    minutes = int(duration/60)
    seconds = int(duration%60)
    hours = int(minutes/60)
    time = (hours, minutes, seconds)
    
    return time
    
def time_calculator(row_number, oa_fps):

    print("oa_fps type: ", type(oa_fps))
    print("oa_fps value: ", oa_fps)

    seconds = int(row_number/oa_fps) % (24 * 3600)
    hour = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
  
    return (hour, minutes, seconds)

# formatting time to print in logger
def time_formatter(time):
    (hours, minutes, seconds) = time
    return (str(hours) + ':' + str(minutes) + ':' + str(seconds))


# index between 0-13
def index_to_object(index):
    object_array = ["hole1", "hole2", "hole3", "hole4", "hole5", "hole6", "carrier", "box1", "box2", "box3", "box4",  "box5", "box6", "exploring"]
    
    return object_array[index]

        
def sort_by_time_func(time_list):
    splited  = [ int(x) for x in time_list[2].split(':')]
    return (splited[0],splited[1], splited[2])    
    

def sort_log_by_time(directory, log_path, video_name):

    f = open(log_path, 'r')
    sorted_log_path = directory + "/sorted_log_file" + "_" + video_name + ".csv"
        
    reader = list(csv.reader(f))
    headers = reader[:1]
    print(headers)
    orig_csv_without_headers = reader[1:]
    print(orig_csv_without_headers)
    sorted_csv = sorted(orig_csv_without_headers, key=sort_by_time_func)
    print(sorted_csv)
    with open(sorted_log_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers[0])
        writer.writerows(sorted_csv)
    
    # remove old log file 
    delete_if_exist(log_path)
    

# receives index and return the right element
def log_decoder(start_row, end_row, oa_fps, index, in_row = 0, out_row = 0):
    start_time = time_calculator(start_row, oa_fps)
    end_time = time_calculator(end_row, oa_fps)
    
    print("index: ", index)
    obj = index_to_object(index)
    
    print('start row:' , start_row)
    print('start time (H:M:S) = %d:%d:%d' % start_time)
    print('end_row: ' , end_row)
    print('end time (H:M:S) = %d:%d:%d' % end_time)
    print('object: %s' % obj)
    
    start_time = time_formatter(start_time)
    end_time =  time_formatter(end_time)
    
    if in_row or out_row:
        in_time = time_calculator(in_row, oa_fps)
        out_time = time_calculator(out_row, oa_fps)
        print('in time (H:M:S) = %d:%d:%d' % in_time)
        print('out time (H:M:S) = %d:%d:%d' % out_time)
        in_time = time_formatter(in_time)
        out_time =  time_formatter(out_time)
        return (obj, start_time, end_time, in_time, out_time)

    return (obj, start_time, end_time)

def logger(f, start_row, end_row, oa_fps, index, event, in_row = 0, out_row = 0):
    if in_row or out_row:
        obj, start_time, end_time, in_time, out_time = log_decoder(start_row, end_row, oa_fps, index, in_row, out_row)
        f.write(str(event) + ',' + str(obj) + ',' + str(start_time) + ',' + str(end_time) +  ',' + str(in_time) + ',' + str(out_time) +'\n')
    else:
        obj, start_time, end_time = log_decoder(start_row, end_row, oa_fps, index)
        f.write(str(event) + ',' + str(obj) + ',' + str(start_time) + ',' + str(end_time) + '\n')
    
    return
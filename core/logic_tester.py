import numpy as np
import core.logic_utils as logic_utils
import core.logger_utils as logger_utils
import csv

def csv_to_event_matrix(file_path):
    
    event_matrix = np.genfromtxt(file_path, delimiter=',')
    return event_matrix


def main():
    
    video_name = "bats2"
    file_path = "C:\\Users\\IdanGursky\\Project\\tensorflow-yolov4-tflite\\detections\\"
    full_path = file_path + "data_" + video_name + ".csv"
    
    event_matrix = csv_to_event_matrix(full_path)
    # remove the first row of headers 
    event_matrix = np.delete(event_matrix, (0), axis=0)
    oa_fps = 14.287897986176995
    test_flag = 1 
    
    ### logger init ###
    log_path = "./detections/log_file"
    log_path, logger = logger_utils.logger_init(log_path, video_name)  
    
    logic_utils.events_logger(logger, event_matrix, oa_fps, log_path, video_name)


    
if __name__ == '__main__':
    main()


import time
import csv
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
import core.object_utils as obj_utils
import core.event_utils as event_utils
import core.logic_utils as logic_utils
import core.logger_utils as logger_utils
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_string('weights_env', './checkpoints/yolov4-416',
                    'path to enviorment weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_string('data_out', './detections/data', 'path to output data')          

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=20)
   

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 416 #input size for enviorment detection
    video_path = FLAGS.video

    # Load enviorment model
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights_env)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights_env, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None
    frame_cnt = 0
    frame_q = [0]
    oa_fps = vid.get(cv2.CAP_PROP_FPS)
    print("oa_fps: ", oa_fps)

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    # read the first frame and register the boxes' locations#######################################################################################################
    env_coor = [0];
    
    ### output files init ###   
    video_name = logger_utils.split_video_name(FLAGS.video)
    xy_path = "./detections/bat_XY"
    log_path = "./detections/log_file"
    event_matrix_file, xy_file, log_path, logger = logger_utils.output_init(FLAGS.data_out, xy_path, log_path, video_name)                            
    while True: # Verifying 6 boxes and 6 holes

        return_value, frame = vid.read()
        frame_cnt += 1
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data) 
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image, env_coor = utils.draw_bbox(frame, pred_bbox, "env", "env") #Added directory of the file containing the classes of the enviorment
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        ### Coor is a 3D array with the coordinates of the two corners of each detection box, for each object and its class name
        boxes = 0;
        holes = 0;
        #######        for env in env_coor:
             ### print("%d at Location: %d, %d, %d, %d; Score: %.2f" % (env[0], env[1][0], env[1][1], env[1][2], env[1][3], env[2]))

        finished, box_list, holes_list = obj_utils.env_established(env_coor)
        if finished:
            event_matrix = event_utils.event_matrix_initializer()
            env_classes = classes
            print("env established")
            
            # Exporting enviorment coordinates
            for i in range(len(box_list)):
                coor = box_list[i]
                xy_file.write('Box ' + str(i+1) + '\n')
                xy_file.write(str(coor[1]) + ',' + str(coor[0]) + ',' + str(coor[3]) + ',' + str(coor[2]) + '\n')

            for i in range(len(holes_list)):
                coor = holes_list[i]
                if i == 6:
                    bbox_mess = '%s' % ("carrier")
                    xy_file.write('Carrier ' + str(i+1) + '\n')
                else: 
                    xy_file.write('Hole ' + str(i+1) + '\n')
                    xy_file.write(str(coor[1]) + ',' + str(coor[0]) + ',' + str(coor[3]) + ',' + str(coor[2]) + '\n') 
            
            break

  
            # result = np.asarray(image)
            # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            # result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    xy_file.write('Frame, Elapsed Time, XTL, YTL, XBR, YBR\n')    
        
    # box_list, holes_list, event_matrix created
    
    # if not FLAGS.dont_show:
        # cv2.imshow("result", result)
    
    # if FLAGS.output:
        # out.write(result)
        
    # reset the weights and the other parameters required for detecting bats ahead ############################################################################
    input_size = FLAGS.size
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
    
    # Going over the rest of the frames #######################################################################################################################
    while True:   
        return_value, frame = vid.read()
        frame_cnt += 1
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
            
        #Motion Detection
        roi = frame[60:1080,:]

        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(mask, None, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        thresh_ind = 0
        motion_frame = frame
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 700:
                continue
            thresh_ind += 1
            
        if thresh_ind == 0 and sum(frame_q) < 5:
            print ('No motion has been detected, skipping frame.')
            image = np.asarray(frame)
            event_matrix = event_utils.event_matrix_updater(event_matrix_file,event_matrix, box_list, holes_list)    
        else:
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            if FLAGS.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data) 
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            image, coor = utils.draw_bbox(frame, pred_bbox, "sub", "sub")
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            ### Coor is a 3D array with the coordinates of the two corners of each detection box, for each object and its class name
           
            ##for xy in coor:
              ##  print("%d at Location: %d, %d, %d, %d; Score: %.2f" % (xy[0], xy[1][0], xy[1][1], xy[1][2], xy[1][3], xy[2]))
                            
            if coor: ### bat is found in the frame 
                print("bat found")
                found = 1 
                frame_q.append(1)
                if len(frame_q) == 7: frame_q.pop(0)
                bat_coor = coor[0][1]
                xy_file.write(str(frame_cnt) + ',' + str(frame_cnt/oa_fps) + ',' + str(bat_coor[1]) + ',' + str(bat_coor[0]) + ',' + str(bat_coor[3]) + ',' + str(bat_coor[2]) + '\n')
                event_matrix = event_utils.event_matrix_updater(event_matrix_file,event_matrix, box_list, holes_list, bat_coor, found)
            else:    ### bat isn't found in the frame
                event_matrix = event_utils.event_matrix_updater(event_matrix_file,event_matrix, box_list, holes_list)
                frame_q.append(0)
                if len(frame_q) == 7: frame_q.pop(0)
            
        ## show box and holes:    
        image_h, image_w, _ = image.shape
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        fontScale = 0.5
        box_color = (0,0,255)
        hole_color = (0, 255, 0)

        for i in range(len(box_list)):
            coor = box_list[i]
            c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
            image = cv2.rectangle(image, c1, c2, box_color, bbox_thick)
            bbox_mess = '%s' % ("box{}".format(i+1))
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), box_color, -1) #filled
            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

        for i in range(len(holes_list)):
            coor = holes_list[i]
            c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
            image = cv2.rectangle(image, c1, c2, hole_color, bbox_thick)
            bbox_mess = '%s' % ("hole{}".format(i+1))
            if i == 6:
                bbox_mess = '%s' % ("carrier")
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), hole_color, -1) #filled
            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
        
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
           
        if not FLAGS.dont_show:
            cv2.imshow("result", result)
        
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    logic_utils.events_logger(logger, event_matrix, oa_fps, log_path, video_name)
    cv2.destroyAllWindows()    
    xy_file.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

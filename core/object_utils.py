import cv2
import numpy as np

# coor = (x1,y1,x2,y2), ratio_limit [0,1]
def intersection_larger_then_limit(coor1, coor2, ratio_limit):
    ############## might need adjustments based on the coor lists structure ###########
    XA1, YA1, XA2, YA2 = coor1
    XB1, YB1, XB2, YB2 = coor2
    
    # compute the intersection, which is a rectangle too
    SIntersection = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
    
    # compute the two rectangles area
    SA = (XA2 - XA1) * (YA2 - YA1)
    SB = (XB2 - XB1) * (YB2 - YB1)
    
    # compute the area of the union
    SUnion = SA + SB - SIntersection
    ratio = SIntersection / SUnion

    return ratio > ratio_limit
    
# debug function to print out objects on top of image
def debug_printer(image, env_coor):

    box_list = []
    holes_list = []
    carrier_list = []    
    
    for env in env_coor:
        if env[0] == 0:
            box_list.append(env)
        elif env[0] == 1:
            holes_list.append(env)
        elif env[0] == 2:
            carrier_list.append(env)
        else:
            print("Class index error")    
    
    # return only the coor of each list
    box_coor = [env[1] for env in box_list]
    holes_coor = [env[1] for env in holes_list]
    carrier_coor = [env[1] for env in carrier_list]

    holes_coor.append(carrier_coor[0])

    ## show box and holes:    
    image_h, image_w, _ = image.shape
    bbox_thick = int(0.6 * (image_h + image_w) / 600)
    fontScale = 0.5
    box_color = (0,0,255)
    hole_color = (0, 255, 0)

    for i in range(len(box_coor)):
        coor = box_coor[i]
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        image = cv2.rectangle(image, c1, c2, box_color, bbox_thick)
        bbox_mess = '%s' % ("box{}".format(i+1))
        t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
        c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), box_color, -1) #filled
        cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
                    
    for i in range(len(holes_coor)):
        coor = holes_coor[i]
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        image = cv2.rectangle(image, c1, c2, hole_color, bbox_thick)
        bbox_mess = '%s' % ("hole{}".format(i+1))
        t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
        c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), hole_color, -1) #filled
        cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    
    result = np.asarray(image)
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("result", result)
    cv2.waitKey(1)
        
       
# receives a coor list of a certain class and check for overlapping 
def check_overlap(class_list):

    # env_list = [(class_ind,coor,score),(class_ind,coor,score), ...]
    # env_list[i][0] = class
    # env_list[i][1] = coor_list[x1,x2,y1,y2]
    # env_list[i][2] = score
        
    overlap_marker = [0] * len(class_list)
    ratio_limit = 0.01
      
    for i in range(len(class_list)):
        for j in range(i + 1, len(class_list)):
            coor_i = class_list[i][1]
            coor_j = class_list[j][1]
            # checking for overlapping elements in the same class
            if intersection_larger_then_limit(coor_i, coor_j, ratio_limit):
                # if they overlap, mark the one that has a lower score
                score_i = class_list[i][2]
                score_j =  class_list[j][2]
                if score_i > score_j:
                    overlap_marker[j] = 1
                else: 
                    overlap_marker[i] = 1
                    

    for i in range(len(overlap_marker))[::-1]:
        if overlap_marker[i] == 1:
            print("overlap detected in index: ", i)
            class_list.pop(i)
        
    return class_list


# class numbers - 0 box, 1 hole , 2 carrier
def class_divider(env_list):
    box_list = []
    holes_list = []
    carrier_list = []    
    
    for env in env_list:
        if env[0] == 0:
            box_list.append(env)
        elif env[0] == 1:
            holes_list.append(env)
        elif env[0] == 2:
            carrier_list.append(env)
        else:
            print("Class index error")    
    
    checked_box_list = check_overlap(box_list)
    checked_holes_list = check_overlap(holes_list)
    checked_carrier_list = check_overlap(carrier_list)
    
    # return only the coor of each list
    box_coor = [env[1] for env in checked_box_list]
    holes_coor = [env[1] for env in checked_holes_list]
    carrier_coor = [env[1] for env in checked_carrier_list]

    return box_coor, holes_coor, carrier_coor
    
    
def remap_lists(box_list,holes_list):
    
    '''
    new mappings
        box1 -> box5 [0 > 4]
        box2 -> box3 [1 > 2]
        box3 -> box1 [2 > 0]
        box4 -> box2 [3 > 1]   
        box5 -> box4 [4 > 3]
        box6 -> box6 [5 > 5]
    '''
    print("box_list: " , box_list)
    print("len of box_list: " , len(box_list))
    print("holes_list: " , holes_list)
    print("len of holes_list: " , len(holes_list))
    
    new_order = [2, 3, 1, 4, 0, 5]
    remapped_box_list = [box_list[i] for i in new_order]
    remapped_holes_list = [holes_list[i] for i in new_order]
    
    return remapped_box_list, remapped_holes_list
    
  
def sort_lists(env_list):

    box_list, holes_list, carrier_list = class_divider(env_list)
    
    # sort holes and boxes lists in a certain order
    box_list = sorted(box_list , key=lambda k: k[1])
    holes_list = sorted(holes_list , key=lambda k: k[1])
        
    return box_list, holes_list, carrier_list


# the only exposed functioמ
# checks if we got exactly 6 boxes and 7 holes (including carrier)
def env_established(env_list):
    box_list, holes_list, carrier_list = sort_lists(env_list)
    
    box_num = len(box_list)
    holes_num = len(holes_list)
    carrier_num = len(carrier_list)
    finished = (box_num == 6 and holes_num == 6 and carrier_num == 1)
    
    if finished:
        # readjusting box and holes list to match current numbering
        box_list, holes_list = remap_lists(box_list,holes_list)
        
        # adding the carrier to the holes at the last spot
        holes_list.append(carrier_list[0])
            
        return finished, box_list, holes_list
    
    return finished, 0, 0


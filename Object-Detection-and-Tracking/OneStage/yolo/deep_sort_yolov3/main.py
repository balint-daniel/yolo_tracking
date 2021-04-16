#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from tensorflow.keras import backend

backend.clear_session()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input video", default = "./test_video/test.avi")
ap.add_argument("-c", "--class",help="name of class", default = "person")
ap.add_argument("-a", "--angle", type=int, default=90,
    help="optionally change the angle of the counter line")
ap.add_argument("-lr", "--leftOrRight", type=float, default=-1.0,
    help="optionally move the vertical line left or right")
ap.add_argument("-ud", "--topOrBottom", type=float, default=-1.0,
    help="optionally move the horizontal line up or down")
args = vars(ap.parse_args())

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

#real_counter = 55 #if we want to calculate the error

def main(yolo):
    left_counter = 0
    right_counter = 0
    already_found = []  # list of the IDs that we've already found
    df_save = pd.DataFrame(columns=['personId', 'direction','every_person_on_the_actual_frame'])  # df containing the intersects
    input_name = args['input']
    input_name = input_name.split('/')[-1]
    input_name = input_name.split('.')[0]  # save the df based on the input file name
    new_intersection = False

    start = time.time()
    #Definition of the parameters
    max_cosine_distance = 0.5 #余弦距离的控制阈值
    nn_budget = None
    nms_max_overlap = 0.3 #非极大抑制的阈值

    counter = []
    #deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    #video_path = "./output/output.avi"
    video_capture = cv2.VideoCapture(args["input"])

    height_line = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width_line = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)

    # rotate the line in the given angle
    # own defined rotation
    counter_line_angle = args["angle"]
    if (counter_line_angle > 44 and counter_line_angle < 91):
        move = 45 / counter_line_angle
        line = [(int(int(width_line) * move), 0), (int(int(width_line) * (1 - move)), int(height_line))]
    elif (counter_line_angle > 90 and counter_line_angle < 136):
        number = 90 - (counter_line_angle - 90)
        move = 45 / number
        line = [(int(int(width_line) * (1 - move)), 0), (int(int(width_line) * move), int(height_line))]
    elif (counter_line_angle > -1 and counter_line_angle < 45):
        move = (45 / (counter_line_angle + 45)) - 0.5
        line = [(int(width_line), int(int(height_line) * move)), (0, int(int(height_line) * (1 - move)))]
    elif (counter_line_angle > 135 and counter_line_angle < 181):
        number1 = 135 - (counter_line_angle - 135)
        number2 = 90 - (number1 - 90)
        move = 45 / number2
        line = [(0, int(int(height_line) * (1 - move))), (int(width_line), int(int(height_line) * move))]
    else:
        raise Exception('Wrong input number for angle!') from None

    # move the vertical line left or right
    # if the "leftOrRight" value is 1.0 then the line is on the right hand side of the video
    # if the "leftOrRight" value is 0.0 then the line is on the left hand side of the video
    # if the "leftOrRight" value is 0.5 then the vertical line is on the center of the video
    # etc.
    left_right = args["leftOrRight"]  # for vertical lines
    if left_right <= 1.0 and left_right >= 0.0 and counter_line_angle >= 45 and counter_line_angle <= 135:  # if its not default
        x1 = line[0][0]
        x2 = line[1][0]
        diff_x = x1 - x2  # save the distance between the points
        line = [(int(int(width_line) * left_right), 0), (int(int(width_line) * left_right - diff_x), int(height_line))]

    # move the horizontal line up or down
    # if the "topOrBottom" value is 1.0 then the line is on the bottom of the video
    # if the "topOrBottom" value is 0.0 then the line is on the top of the video
    # if the "topOrBottom" value is 0.5 then the horizontal line is on the center of the video
    # etc.
    up_down = args["topOrBottom"]  # for horizontal lines
    if (up_down <= 1.0 and up_down >= 0.0 and counter_line_angle > 135 and counter_line_angle <= 180) or \
            (
                    up_down <= 1.0 and up_down >= 0.0 and counter_line_angle >= 0 and counter_line_angle < 45):  # if its not default
        y1 = line[0][1]
        y2 = line[1][1]
        diff_y = y1 - y2
        line = [(line[0][0], int(int(height_line) * up_down)), (line[1][0], int(int(height_line) * up_down - diff_y))]

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('./output/'+args["input"][43:57]+ "_" + args["class"] + '_output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0

    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs,class_names = yolo.detect_image(image)
        features = encoder(frame,boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        c = []
        boxes = []
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)
            if len(class_names) > 0:
               class_name = class_names[0]
               cv2.putText(frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)

            i += 1
            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]
            pts[track.track_id].append(center)
            thickness = 5
            #center point
            cv2.circle(frame,  (center), 1, color, thickness)

	        #draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                   continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
                #cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                   continue
                p0 = (pts[track.track_id][j])  # actual center coords
                p1 = (pts[track.track_id][j - 1]) # previous center coords

                if intersect(p0, p1, line[0], line[1]):
                    if track.track_id not in already_found:
                        new_intersection = True
                        df_temp = pd.DataFrame({'personId': [track.track_id]})  # df with the actual personID
                        if counter_line_angle >= 45 and counter_line_angle <= 135:  # if the line is vertical
                            if p0[0] > p1[0]:  # if they cross the line from left to right
                                left_counter += 1
                                df_temp['direction'] = "left to right"
                                df_temp['actual_left_to_right_counter'] = left_counter
                            else:  # if they cross the line from right to left
                                right_counter += 1
                                df_temp['direction'] = "right to left"
                                df_temp['actual_right_to_left_counter'] = right_counter
                        else:  # if the line is horizontal
                            if p0[1] > p1[1]:  # if they cross the line from top to bottom
                                left_counter += 1
                                df_temp['direction'] = "top to bottom"
                                df_temp['actual_top_to_bottom_counter'] = left_counter
                            else:  # if they cross the line from bottom to top
                                right_counter += 1
                                df_temp['direction'] = "bottom to top"
                                df_temp['actual_bottom_to_top_counter'] = right_counter
                        already_found.append(track.track_id)
                        df_temp['actual_total_person_counter'] = len(set(counter))
                        df_save = df_save.append(df_temp, ignore_index=True)  # append the actual intersection
                        df_save.to_csv(f"output/df_save_{input_name}.csv", index=False)  # save the df

        if(new_intersection):
            new_intersection = False
            df_save['every_person_on_the_actual_frame'].iloc[df_save.shape[0]-1] = i
            cols_at_end = ['every_person_on_the_actual_frame','actual_total_person_counter']
            df_save = df_save[
                [c for c in df_save if c not in cols_at_end] + [c for c in cols_at_end if c in df_save]]
            df_save.to_csv(f"output/df_save_{input_name}.csv", index=False)  # save the df

        # draw the black rectangles at the corners of the video
        cv2.rectangle(frame, (0, int(height_line - height_line / 5)), (int(width_line / 8), int(height_line)),
                      (0, 0, 0), -1)  # left to right
        cv2.rectangle(frame, (int(width_line - width_line / 8), int(height_line - height_line / 5)),
                      (int(width_line), int(height_line)), (0, 0, 0), -1)  # right to left

        # draw counters
        if counter_line_angle >= 45 and counter_line_angle <= 135:
            cv2.putText(frame, str("Left to right:"), (10, int(height_line - height_line / 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, str("Top to bottom:"), (10, int(height_line - height_line / 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, str(left_counter), (20, int(height_line - 30)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
                    5)

        if counter_line_angle >= 45 and counter_line_angle <= 135:
            cv2.putText(frame, str("Right to left:"),
                        (int(width_line - width_line / 9), int(height_line - height_line / 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, str("Bottom to top:"),
                        (int(width_line - width_line / 9), int(height_line - height_line / 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, str(right_counter), (int(width_line - width_line / 11), int(height_line - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

        #draw the main counter line
        cv2.line(frame, line[0], line[1], (0, 255, 0), 10)

        count = len(set(counter))
        #cv2.putText(frame, str("Error: " + str(round(abs(count - real_counter) / real_counter, 2))),(int(20),int(160)), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "Total Object Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "Current Object Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
        cv2.namedWindow("YOLO3_Deep_SORT", 0);
        cv2.resizeWindow('YOLO3_Deep_SORT', 1024, 768);
        cv2.imshow('YOLO3_Deep_SORT', frame)

        if writeVideo_flag:
            #save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        #print(set(counter))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(" ")
    print("[Finish]")
    end = time.time()

    if len(pts[track.track_id]) != None:
       print(args["input"][43:57]+": "+ str(count) + " " + str(class_name) +' Found')

    else:
       print("[No Found]")

    video_capture.release()

    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob

files = glob.glob('output/*.png')
for f in files:
   os.remove(f)

from sort import *
tracker = Sort()
memory = {}
counter = 0
left_counter = 0
right_counter = 0
#real_counter = 18 # if we want to calculate the error.
import math
from collections import deque
import warnings
import pandas as pd

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",
	help="path to input video", default = "./input/det_t1_video_00031_test.avi")
ap.add_argument("-o", "--output",
	help="path to output video", default = "./output/")
ap.add_argument("-y", "--yolo",
	help="base path to YOLO directory", default = "./yolo-obj")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
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

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])

height_line = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
width_line = vs.get(cv2.CAP_PROP_FRAME_WIDTH)

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
	move = ( 45 / (counter_line_angle + 45) ) - 0.5
	line = [(int(width_line), int(int(height_line)*move)), (0, int(int(height_line)*(1-move)))]
elif (counter_line_angle > 135 and counter_line_angle < 181):
	number1 = 135 - (counter_line_angle - 135)
	number2 = 90 - (number1 - 90)
	move = 45 / number2
	line = [(0, int(int(height_line)*(1-move))), (int(width_line), int(int(height_line)*move))]
else:
    raise Exception('Wrong input number for angle!') from None

# move the vertical line left or right
# if the "leftOrRight" value is 1.0 then the line is on the right hand side of the video
# if the "leftOrRight" value is 0.0 then the line is on the left hand side of the video
# if the "leftOrRight" value is 0.5 then the vertical line is on the center of the video
# etc.
left_right = args["leftOrRight"] #for vertical lines
if left_right <= 1.0 and left_right >= 0.0 and counter_line_angle >= 45 and counter_line_angle <= 135: # if its not default
	x1 = line[0][0]
	x2 = line[1][0]
	diff_x = x1 - x2 # save the distance between the points
	line = [(int(int(width_line)*left_right),0),(int(int(width_line)*left_right-diff_x),int(height_line))]

# move the horizontal line up or down
# if the "topOrBottom" value is 1.0 then the line is on the bottom of the video
# if the "topOrBottom" value is 0.0 then the line is on the top of the video
# if the "topOrBottom" value is 0.5 then the horizontal line is on the center of the video
# etc.
up_down = args["topOrBottom"] #for horizontal lines
if (up_down <= 1.0 and up_down >= 0.0 and counter_line_angle > 135 and counter_line_angle <= 180) or \
		(up_down <= 1.0 and up_down >= 0.0 and counter_line_angle >= 0 and  counter_line_angle < 45) : # if its not default
	y1 = line[0][1]
	y2 = line[1][1]
	diff_y = y1 - y2
	line = [(line[0][0],int(int(height_line)*up_down)),(line[1][0],int(int(height_line)*up_down-diff_y))]

writer = None
(W, H) = (None, None)

frameIndex = 0

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

already_found = [] # list of the IDs that we've already found
pts = [deque(maxlen=30) for _ in range(9999)] # for the motion path
warnings.filterwarnings('ignore')
df_save = pd.DataFrame(columns=['personId','direction']) # df containing the intersects
input_name = args['input']
input_name = input_name.split('/')[-1]
input_name = input_name.split('.')[0] # save the df based on the input file name

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	current_counter = 0  # the current number of people on the actual frame

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			#if confidence > args["confidence"]:
			if confidence > args["confidence"] and classID == 0: #detect only people
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)


	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

	dets = []
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			dets.append([x, y, x+w, y+h, confidences[i]])

	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
	dets = np.asarray(dets)
	tracks = tracker.update(dets)

	boxes = []
	indexIDs = []
	c = []
	previous = memory.copy()
	memory = {}


	for track in tracks:
		boxes.append([track[0], track[1], track[2], track[3]])
		indexIDs.append(int(track[4]))
		memory[indexIDs[-1]] = boxes[-1]

	if len(boxes) > 0:
		i = int(0)
		for box in boxes:
			# extract the bounding box coordinates
			(x, y) = (int(box[0]), int(box[1]))
			(w, h) = (int(box[2]), int(box[3]))

			# draw a bounding box rectangle and label on the image
			# color = [int(c) for c in COLORS[classIDs[i]]]
			# cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

			color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
			cv2.rectangle(frame, (x, y), (w, h), color, 8)

			# fill the box
			sub_img = frame[y:int(h), x:int(w)]
			rect = np.zeros(sub_img.shape, dtype=np.uint8)
			res = cv2.addWeighted(sub_img, 0.5, rect, 0.5, 1.0) # fill the box with black color
			frame[y:int(h), x:int(w)] = res

			current_counter += 1  # increase the current number of found people

			if indexIDs[i] in previous:
				previous_box = previous[indexIDs[i]]
				(x2, y2) = (int(previous_box[0]), int(previous_box[1]))
				(w2, h2) = (int(previous_box[2]), int(previous_box[3]))
				p0 = (int(x + (w-x)/2), int(y + (h-y)/2)) # actual center coords
				p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2)) # previous center coords
				cv2.line(frame, p0, p1, color, 3) #draw line between the actual and the previous center coord

				pts[indexIDs[i]].append(p0) # append tha actual center coordinates for drawing the motion path

				# draw motion path
				for j in range(1, len(pts[indexIDs[i]])):
					if pts[indexIDs[i]][j - 1] is None or pts[indexIDs[i]][j] is None:
						continue
					thickness = int(np.sqrt(64 / float(j + 1)) * 2)
					cv2.line(frame, (pts[indexIDs[i]][j - 1]), (pts[indexIDs[i]][j]), (color), thickness)

				if intersect(p0, p1, line[0], line[1]):
					if indexIDs[i] not in already_found: # count only when we find the ID for the first time
						counter += 1
						df_temp = pd.DataFrame({'personId': [indexIDs[i]]}) # df with the actual personID
						if counter_line_angle >= 45 and counter_line_angle <= 135: # if the line is vertical
							if p0[0] > p1[0]: #if they cross the line from left to right
								left_counter += 1
								df_temp['direction'] = "left to right"
								df_temp['actual_left_to_right_counter'] = left_counter
							else: # if they cross the line from right to left
								right_counter += 1
								df_temp['direction'] = "right to left"
								df_temp['actual_right_to_left_counter'] = right_counter
						else: # if the line is horizontal
							if p0[1] > p1[1]: #if they cross the line from top to bottom
								left_counter += 1
								df_temp['direction'] = "top to bottom"
								df_temp['actual_top_to_bottom_counter'] = left_counter
							else: #if they cross the line from bottom to top
								right_counter += 1
								df_temp['direction'] = "bottom to top"
								df_temp['actual_bottom_to_top_counter'] = right_counter
						already_found.append(indexIDs[i])
						df_temp['actual_person_counter'] = counter
						df_temp['every_person_on_the_actual_frame'] = current_counter

						df_save = df_save.append(df_temp, ignore_index=True) # append the actual intersection
						cols_at_end = ['actual_person_counter','every_person_on_the_actual_frame']  # move these columns to the end of the df
						df_save = df_save[[c for c in df_save if c not in cols_at_end] + [c for c in cols_at_end if c in df_save]]
						df_save.to_csv(f"output/df_save_{input_name}.csv",index=False) # save the df

			# text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			text = "{}".format(indexIDs[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 5)
			i += 1

	# draw the main counter line
	cv2.line(frame, line[0], line[1], (0, 255, 0), 10)

	# draw the black rectangles at the corners of the video
	cv2.rectangle(frame, (0, int(height_line-height_line/5)), (int(width_line/8), int(height_line)), (0, 0, 0), -1)  # left to right
	cv2.rectangle(frame, (int(width_line-width_line/8), int(height_line-height_line/5)), (int(width_line), int(height_line)),(0, 0, 0), -1)  # right to left
	cv2.rectangle(frame, (0, 0), (int(width_line/8), int(height_line/5)), (0, 0, 0), -1)  # counter
	cv2.rectangle(frame, (int(width_line-width_line/8), 0), (int(width_line), int(height_line/5)), (0, 0, 0), -1)  # left - right

	# draw counters
	if counter_line_angle >= 45 and counter_line_angle <= 135:
		cv2.putText(frame, str("Left to right:"), (10,int(height_line-height_line/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
	else:
		cv2.putText(frame, str("Top to bottom:"), (10,int(height_line-height_line/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
	cv2.putText(frame, str(left_counter), (20, int(height_line-30)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

	if counter_line_angle >= 45 and counter_line_angle <= 135:
		cv2.putText(frame, str("Right to left:"), (int(width_line-width_line/9),int(height_line-height_line/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
	else:
		cv2.putText(frame, str("Bottom to top:"), (int(width_line-width_line/9),int(height_line-height_line/6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
	cv2.putText(frame, str(right_counter), (int(width_line-width_line/11), int(height_line - 30)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

	cv2.putText(frame, str("Counted people:"), (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
	cv2.putText(frame, str(counter), (20, int(height_line/7)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

	# draw the error:
	#cv2.putText(frame, str("Error: " + str(round(abs(counter-real_counter)/real_counter,2))), (int(width_line/2), 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

	cv2.putText(frame, str("Current people counter: " + str(current_counter)), (int(width_line / 2), 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

	if counter_line_angle >= 45 and counter_line_angle <= 135:
		cv2.putText(frame, str("Left - Right ="), (int(width_line-width_line/9),20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
	else:
		cv2.putText(frame, str("Top - Bottom="), (int(width_line-width_line/9),20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
	cv2.putText(frame, str(left_counter-right_counter), (int(width_line-width_line/11), int(height_line/7)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

	# saves image file
	cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame)

	# increase frame index
	frameIndex += 1

	if frameIndex >= 4000:
		print("[INFO] cleaning up...")
		writer.release()
		vs.release()
		exit()

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()

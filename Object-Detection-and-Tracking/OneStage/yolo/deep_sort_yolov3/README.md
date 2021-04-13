# YOLOv3 + Deep_SORT

## Quick Start
    
__1. Download the code to your computer.__
        
__2. Download [yolov3.weights] (https://pjreddie.com/media/files/yolov3.weights)__ and place it in `deep_sort_yolov3/model_data/`

__3. Convert the Darknet YOLO model to a Keras model:__
```
$ python convert.py model_data/yolov3.cfg model_data/yolov3.weights model_data/yolo.h5
``` 
	-------------------IF it does not run, try:     pip install -r requirements.txt -------------------------

__4. Run the YOLO_DEEP_SORT:__

```
$ python main.py -c [CLASS NAME] -i [INPUT VIDEO PATH]

$ python main.py -c person -i ../input/airport_walking.mp4 --angle 180 --topOrBottom 0.4

```





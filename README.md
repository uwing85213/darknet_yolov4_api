# darknet_yolov4_api


# Tips

GPU：RTX 3090

Cuda：11.1

Cudnn：8.0.5

darknet細節請參照：https://github.com/AlexeyAB/darknet

# Install

```
pip install -r  requirements.txt
```


# Use

```
from YingRen_Yolov4API import Yolov4YingRen
min_score_thresh=0.5
MODEL_PATH = 'model/coco/'
config_file_Path = MODEL_PATH + 'coco.cfg'
data_file_path = MODEL_PATH + 'coco.data'
weights_Path = MODEL_PATH + 'coco.weights'

detector = Yolov4YingRen(config_file_Path , data_file_path , weights_Path)

img , objs = detector.catchObject_J(frame , min_score_thresh , 'ON') 
```


# Example
```
python objdetection_video.py
```

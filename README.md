# darknet_yolov4_api


Install

```
pip install -r  requirements.txt
```


Use

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


Example
```
python objdetection_video.py
```

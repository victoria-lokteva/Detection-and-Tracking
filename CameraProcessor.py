import os
import torch
import numpy as np
from SORT.sort import Sort
from data_structs import  ObjectTracklet, ObjectCrop, TrackPosition
import time
import imageio
import json


class CameraProcessor():
    def __init__(self, output_path="result.json", score_threshold=0.5, model=torch.hub.load('ultralytics/yolov5',
                                                'yolov5s')):
        self.output_path = output_path
        self.score_threshold = score_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.eval().to(self.device)
 
        self.active_tracklets = {}
        self.sorter = Sort()
        self.times = []


    def track(self, video):
        reader = imageio.get_reader(video)
        for i, frame in enumerate(reader):
            started = time.time()
            height, width = frame.shape[:2]
            x = [frame]
#             x = torch.Tensor(x).to(self.device)
            predictions = self.model(x, size=640)
            predictions = predictions.pred[0].to('cpu').numpy()
            tracks = self.sorter.update(predictions)
            tracks = tracks.tolist()
            data = []
            for coords in tracks:
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                name = int(coords[4])
                box = [x1, y1, x2, y2 ]
                bbox = ObjectCrop(video_path=video, frame_id=i, bbox=box)
                track_obj = TrackPosition(x=x, y=y)
                obj = ObjectTracklet(id=name, zone_id=zone_id, trackpath=list(track_obj), crops=list(bbox))
                data.append(obj)
            self.active_tracklets[i] = data
            ended = time.time()
            self.times.append(ended-started)
       

    def result(self):
        with open("result.json", 'w') as file:
            json.dump(self.active_tracklets, file)

    def open(self, path):
        with open(path, 'r') as file:
            data = json.load( file)
        return data


    def count_percentile(self):
        print(f"90 % percentile {np.percentile(self.times, 90)} ms per box ")
        print(f"95 % percentile {np.percentile(self.times, 95)} ms per box ")
        print(f"100 % percentile {np.percentile(self.times, 100)} ms per box ")

import json
import fire
import imageio
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class Tracker():
    def __init__(self, video_path='traffic2.mp4', output_path= "result.json", score_threshold = 0.5,
                 class_index = 1, model='fasterrcnn'):
        self.video_path = video_path
        self.output_path = output_path
        self.score_threshold = score_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model == 'fasterrcnn':
            self.model = fasterrcnn_resnet50_fpn(pretrained=True).eval().to(self.device)
        elif model == 'yolo':
            self.model = torch.hub.load('ultralytics/yolov5',
                                                'yolov5s').eval().to(self.device)
        else:
            raise Exception("Incorrect model name")
        self.reader = imageio.get_reader(video_path)
        self.active_tracklets = []
        self.finished_tracklets = []
        self.prev_boxes = []
        
        
    def track(self):
        for i, frame in enumerate(self.reader):
            height, width = frame.shape[:2]
            x = to_tensor(frame).to(self.device)
            result = self.model(x[None])[0]
            # (x1, y1, x2, y2) - image coordinates
            mask = torch.logical_and(result["labels"] == self.class_index,
                                     result["scores"] > self.score_threshold)
            boxes = result["boxes"][mask].data.cpu().numpy() / np.array(
                                    [width, height, width, height])
            prev_indices = []
            boxes_indices = []
            if len(boxes) * len(self.prev_boxes) > 0:
                cost = np.linalg.norm(self.prev_boxes[:, None] - boxes[None], axis=-1)
                prev_indices, boxes_indices = linear_sum_assignment(cost)

            for prev_idx, box_idx in zip(prev_indices, boxes_indices):
                self.active_tracklets[prev_idx]["boxes"].append(np.round(boxes[box_idx], 
                                                                         3).tolist())

            lost_indices = set(range(len(self.active_tracklets))) - set(prev_indices)
            for lost_idx in sorted(lost_indices, reverse=True):
                 self.finished_tracklets.append(self.active_tracklets.pop(lost_idx))
            new_indices = set(range(len(boxes))) - set(boxes_indices)
            for new_idx in new_indices:
                self.active_tracklets.append(
                    {"start": i, "boxes": [np.round(boxes[new_idx], 3).tolist()]})
            self.prev_boxes = np.array([tracklet["boxes"][-1] for tracklet
                                   in self.active_tracklets])
            
    def save(self):        
        with open(self.output_path, "w") as f:
            f.write(json.dumps({"fps": self.reader.get_meta_data()["fps"],
                    "tracklets": self.finished_tracklets + self.active_tracklets}))
            
if __name__ == "__main__":
    tracker = Tracker()
    fire.Fire(tracker.track())

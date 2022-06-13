import json
import fire
import time
import imageio
import numpy as np
import torch
import ast
from scipy.optimize import linear_sum_assignment
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor


def track( video_path='traffic2.mp4', output_path= "result.json", score_threshold = 0.5,
          class_index = 1, measure_time=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s').eval().to(device)
    reader = imageio.get_reader(video_path)
    times = []
    active_tracklets = []
    finished_tracklets = []
    prev_boxes = []
    for i, frame in enumerate(reader):
        started = time.time()
        height, width = frame.shape[:2]
        # x = to_tensor(frame).to(device)
        # x = x.unsqueeze(0)
        x= [frame]
        results = model(x, size=640)
        results = results.pandas().xyxy[0].to_json(orient="records") 
        result = dict()
        result["labels"] = []
        result["scores"] = []
        result["boxes"] = []
        results = ast.literal_eval(results)
        for item in results:
          result["scores"].append(item["confidence"])
          result["labels"].append(item["class"])
          box = [item['xmin'], item['ymin'], item['xmax'], item['ymax']]
          result["boxes"].append(box)
        result["labels"] = torch.tensor(result["labels"] )
        result["scores"] = torch.tensor(result["scores"] )
        result["boxes"] = torch.tensor(result["boxes"] )

        # (x1, y1, x2, y2) - image coordinates
        mask = torch.logical_and(result["labels"] == class_index,
                                 result["scores"] > score_threshold)
        boxes = result["boxes"][mask].data.cpu().numpy() / np.array(
                                [width, height, width, height])
        prev_indices = []
        boxes_indices = []
        if len(boxes)*len(prev_boxes) > 0:
            cost = np.linalg.norm(prev_boxes[:, None] - boxes[None], axis=-1)
            prev_indices, boxes_indices = linear_sum_assignment(cost)

        for prev_idx, box_idx in zip(prev_indices, boxes_indices):
            active_tracklets[prev_idx]["boxes"].append(np.round(boxes[box_idx], 3).tolist())
            
        lost_indices = set(range(len(active_tracklets))) - set(prev_indices)
        for lost_idx in sorted(lost_indices, reverse=True):
            finished_tracklets.append(active_tracklets.pop(lost_idx))
        new_indices = set(range(len(boxes))) - set(boxes_indices)
        for new_idx in new_indices:
            active_tracklets.append(
                {"start": i, "boxes": [np.round(boxes[new_idx], 3).tolist()]}
            )
        prev_boxes = np.array([tracklet["boxes"][-1] for tracklet in active_tracklets])
        ended = time.time()
        if measure_time and i%50==0:
           times.append(ended-started)

    if measure_time:
       print("Average inference time per frame: ", sum(times)/len(times))


    with open(output_path, "w") as f:
        f.write(json.dumps({"fps": reader.get_meta_data()["fps"],
                "tracklets": finished_tracklets + active_tracklets}))
      

if __name__ == "__main__":
    fire.Fire(track)

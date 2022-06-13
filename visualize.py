import json
import torchvision
import torch
import imageio
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def format_img(image):
    """PIL Image to tensor [C, H, W]"""
    
    frame = np.array(image)
    frame = frame.swapaxes(0,2)
    frame = frame.swapaxes(1,2)
    frame = torch.tensor(frame,  dtype=torch.uint8)
    return frame

    
def format_box(box):
    box = box[2:6]
    box[2] = box[0] + box[2]
    box[3] = box[1] + box[3]
    cur = torch.tensor(box,  dtype=torch.float32)
    return cur
    
    
def draw_boxes(video, labels):
    """video - list of PIL images, labels - list of bounding boxes"""
    reader = iter(video) 
    labels = iter(labels)
    cur_box = None
    frames = []
    BB = []
    for i, frame in enumerate(reader):
        frame = format_img(frame)
        _, h, w = frame.shape
        if cur_box is None:
            cur_box = next(labels)
        while cur_box[0] == i+1:
            cur_box = format_box(cur_box)
            cur = torch.tensor(cur_box,  dtype=torch.int32)
            cur_box = cur_box.unsqueeze(0)
            frame = draw_bounding_boxes(frame, cur_box, width=5, colors=(0,0,255))
            try:
                cur_box = next(labels)
            except(StopIteration):
                break
        img = frame.clone().detach()
        img = torchvision.transforms.ToPILImage()(img)
        frames.append(img)
        BB.append(cur_box)   
    # video = cv2.VideoWriter("tracking.mp4", 0, 1,
    #                         (frame.shape[0],frame.shape[1]))
    # for image in frames:
    #   video.write(cv2.imread(image))
    # cv2.destroyAllWindows()
    # video.release()
    return frames, BB

frames, BB = draw_boxes(video, labels)

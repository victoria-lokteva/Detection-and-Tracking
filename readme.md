## Dataset

At first it was used TAO Dataset, containing videos of people in the shopping malls and streets and corresponding bounding boxes

https://motchallenge.net/tao_download.php

Then it was used videodata at mp4 format, videodata was downloaded from this resource

https://www.istockphoto.com/search/2/film?phrase=people&irgwc=1&cid=IS&utm_medium=affiliate&utm_source=Oxford%20Media%20Solutions&clickid=Q-Jx4GXL6xyIT8F2vITtFy6IUkD2YE0xHUPDws0&utm_term=people&utm_campaign=red_indirect_all&utm_content=1020584&irpid=51471


## Inference

python tracker.py <video_name> --class-index <...>

## Results

The inference was performed in Colab

The inference time (ms per box)

|Model | 100%  |  95%   |  90%   |  50%  |
|------|-------|--------|--------|-------|
| FRCNN| 0.62  |  0.43  |  0.40  | 0.39  |
|YOLOv5| 0.056 | 0.050  | 0.047  | 0.043 |

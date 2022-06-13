import os
from torch.utils.data import Dataset
from PIL import Image

class VDataset(Dataset):
    
    def __init__(self, path, transform=None):
        "TAO dataset format (MOT16)"
        super().__init__()
        self.transform = transform
        self.path = path
        self.videos = sorted(os.listdir(path))
        self.videos = [v for v in self.videos if not v.startswith('.')]
        self.idx = -1
        self.used = 0
        self.prev_used = 0
        self.images = []
        
    def __len__(self):
        return len(self.videos)
    
    def new_video(self):
        self.prev_used = self.used
        self.images = []
        file = self.videos[self.idx]
        file = os.path.join(self.path, file)
        img_file = os.path.join(file, 'img1')
        desc_file = os.path.join(file, 'det/det.txt')
        with open(desc_file, 'r') as f:
            lines = f.readlines()
        lines = [(line.strip('\n')).split(',') for line in lines]
        self.n_lines = []
        for l in lines:
            l = [float(el)  if el.find('.')!=-1 else int(el) for el in l]
            self.n_lines.append(l)
        frames = os.listdir(img_file)
        frames= [v for v in frames if not v.startswith('.')]
        
        self.used += len(frames)
        frames = sorted(frames)

        for frame in frames:
            frame = os.path.join(img_file, frame)
            img = Image.open(frame)
            if transform:
                img = self.transform(img)
            self.images.append(img)
        
    def __getitem__(self, idx):
        idx -= self.prev_used
        if idx == 0 or idx >= len(self.images)-1:
            self.idx+=1
            self.new_video()
        idx -= self.prev_used
        return self.images[idx], self.n_lines[idx]

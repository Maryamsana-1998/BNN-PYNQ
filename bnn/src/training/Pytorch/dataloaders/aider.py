import os
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

"""Aerial Imagery Dataset for Emergency Response"""


class AIDER(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.as_tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(Image.fromarray(image))

        return image, y_label


aider_transforms = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.CenterCrop(240),
    transforms.ToTensor()])
squeeze_transforms = transforms.Compose([
    transforms.Resize((140, 140)),
    transforms.CenterCrop(140),
    transforms.ToTensor()])

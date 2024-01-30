from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import os
import random
import torchvision.transforms.functional as F

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(),  # Use only if the input image is not a PIL image
                                                 tv.transforms.ToTensor(),
                                                 tv.transforms.Normalize(mean=train_mean, std=train_std)])

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transformation):
        self._transform = tv.transforms.Compose(transformation)

    def __len__(self):
        return len(self.data) * 2

    def shuffle(self):
        # Shuffle the DataFrame
        self.data = self.data.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        # Determine if the requested index is for an original image or its flipped version
        original_index = index % len(self.data)  # Find the corresponding index in the original dataset
        flip = index >= len(self.data)  # Determine if the image should be flipped

        # Load labels
        sample = self.data.iloc[original_index]
        relative_path = sample["filename"]
        crack_label = sample["crack"]
        inactive_label = sample["inactive"]

        # Load image
        script_directory = os.path.dirname(os.path.abspath(__file__))
        # Construct the absolute path to the image
        absolute_path_to_image = os.path.join(script_directory, relative_path)
        image = imread(absolute_path_to_image)

        # Convert to RGB if grayscale
        image = gray2rgb(image)

        # Conditionally apply the horizontal flip transformation
        if flip:
            image = F.hflip(image)

        # Apply other transformations
        image = self.transform(image)

        # Stack labels into tensor
        labels = torch.tensor([crack_label, inactive_label])

        return image, labels

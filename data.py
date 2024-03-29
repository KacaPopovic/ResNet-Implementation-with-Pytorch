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
        return len(self.data)

    def shuffle(self):
        # Shuffle the DataFrame
        self.data = self.data.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        # Determine if the requested index is for an original image or its flipped version
        # Load labels
        sample = self.data.iloc[index]
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

        # TODO add different transforms for training
        # Perform transformations on the image.
        if self.mode == 'train1':
          self._transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.RandomApply([tv.transforms.RandomRotation((90, 90)),
                                       tv.transforms.RandomRotation((180, 180)),
                                       tv.transforms.RandomRotation((270, 270))], p=0.2),
            tv.transforms.RandomVerticalFlip(p=0.5),
            tv.transforms.ToTensor(),
            tv.transforms.RandomErasing()
        ])
        else:
          self._transform = tv.transforms.Compose([
              tv.transforms.ToPILImage(),
              tv.transforms.ToTensor(),
              tv.transforms.Normalize(mean=train_mean, std=train_std)
          ])
        # Apply other transformations
        image = self.transform(image)

        crack_label = int(crack_label) if isinstance(crack_label, str) else crack_label
        inactive_label = int(inactive_label) if isinstance(inactive_label, str) else inactive_label

        # Stack labels into tensor
        labels = torch.tensor([crack_label, inactive_label])

        return image, labels

"""
Maps dataset link: https://www.kaggle.com/vikramtiwari/pix2pix-dataset
Anime dataset link: https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair
"""

from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import config

class MapDataset(Dataset):
    def __init__(self, root_dir):
        """

        Args:
            root_dir ():
        """
        super().__init__()
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        # print(f"Files: {self.list_files}")

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, item):
        img_file = self.list_files[item]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        augmentations = config.both_transform(image = input_image, image0 = target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]

        input_image = config.transform_only_input(image = input_image)["image"]
        target_image = config.transform_only_input(image = target_image)["image"]

        return input_image, target_image
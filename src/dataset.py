import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
from config import ROOT_DIR_TEST
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ShipDataset(Dataset):
    def __init__(self, root_dir, csv_file):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file, index_col=0)

    def __len__(self):
        return len(self.df)

    def _rle_to_mask(self, rle, width=768, height=768):
        mask = np.zeros(width * height, dtype=np.uint8)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[::2]
        lengths = array[1::2]
        for start, length in zip(starts, lengths):
            mask[start : start + length] = 1
        return mask.reshape((height, width), order="F")

    def _load_masks(self, image_name):
        image_row = self.df.loc[self.df["ImageId"] == image_name]
        if not pd.isnull(image_row.iloc[0]["EncodedPixels"]):
            temp_masks = []
            for rle in image_row["EncodedPixels"]:
                mask = self._rle_to_mask(rle)
                temp_masks.append(mask)
            masks = np.sum(temp_masks, axis=0)
            return masks / 1.0
        else:
            # Return an empty mask if there are no positive masks for the image
            # will be used during fine-tuning on full-sized images and inference
            return np.zeros((768, 768)) / 1.0

    def _load_images(self, image_name):
        # Check if the folder path exists
        if not os.path.exists(self.root_dir):
            raise ValueError(f"The folder path '{self.root_dir}' does not exist.")

        if os.path.isfile(os.path.join(self.root_dir, image_name)):
            # Open the image file using PIL with the 'with' statement
            with Image.open(os.path.join(self.root_dir, image_name)) as image:
                # copy image
                image = image.copy()

        return image

    def __getitem__(self, index):
        image_name = self.df.iloc[index]["ImageId"]
        image = np.array(self._load_images(image_name)).astype(np.float32) / 255.0
        mask = (
            np.array(
                torch.tensor(self._load_masks(image_name=image_name))
                .unsqueeze(0)
                .float()
            )
            / 1.0
        )

        return torch.Tensor(image), torch.Tensor(mask)


# augmentation for test
TRANSFORM_VAL_TEST = A.Compose([ToTensorV2()])


# Load and preprocess the input image
def preprocess_image(image_name):
    """
    input: image name
    output: torch.Tensor image
    """
    # Check if the folder path exists
    if not os.path.exists(ROOT_DIR_TEST):
        raise ValueError(f"The folder path '{ROOT_DIR_TEST}' does not exist.")
    if os.path.isfile(os.path.join(ROOT_DIR_TEST, image_name)):
        # Open the image file using PIL with the 'with' statement
        with Image.open(os.path.join(ROOT_DIR_TEST, image_name)) as image:
            image = np.array(image).astype(np.float32) / 255.0
            aug = TRANSFORM_VAL_TEST(image=image)
            preprocessed = aug["image"]
            return preprocessed


class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.image_files = [
            file
            for file in os.listdir(root_dir)
            if file.endswith((".jpg", ".png", ".jpeg"))
        ]
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        img = preprocess_image(image_name)
        return img, image_name

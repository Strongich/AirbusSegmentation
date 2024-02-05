import torch
import os
from config import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
import pandas as pd
from tqdm import tqdm
from dataset import TestDataset
from torch.utils.data import DataLoader
from get_model import download_file_if_empty

device = DEVICE


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def main():
    # create folder for output .csv file
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    # check if the model is downloaded locally
    download_file_if_empty(SAVED_MODEL_PATH, GDRIVE_LINK, "model.pth")
    dataset = TestDataset(ROOT_DIR_TEST)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
    # Load the model state dictionary
    checkpoint = torch.load(SAVED_MODEL_PATH + "model.pth", map_location=DEVICE)
    # define model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    # load our local model
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    # lists for .csv file
    filenames = []
    output = []
    for batch in tqdm(dataloader, desc="Processing Images"):
        image, image_name = batch
        image = image.to(DEVICE)
        with torch.no_grad():
            # get mask prediction
            prediction = model(image)
            # convert to binary: 1 - mask, 0 - background
            prediction = (
                (torch.sigmoid(prediction) > 0.6).cpu().numpy().astype(np.uint8)
            )
            for pred in prediction:
                output.append(rle_encode(pred))
            filenames.extend(image_name)
    # create output file
    df = pd.DataFrame({"ImageId": filenames, "EncodedPixels": output})
    df.to_csv(OUTPUT_PATH + "output.csv")
    print(f"Check {OUTPUT_PATH} for result")


if __name__ == "__main__":
    main()

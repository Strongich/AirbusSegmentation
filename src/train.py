from torch.utils.data import DataLoader
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch
import math
import argparse
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from loss import mixed_dice_bce_loss
from config import (
    DEVICE,
    SAVED_MODEL_PATH,
    ROOT_DIR_TRAIN,
    LABEL_PATH,
    LOAD_MODEL,
)

EPOCHS = 10
PATH_TO_MODEL = "../models/model.pth"
from dataset import ShipDataset


# 1 train loop
def train_loop(model, optimizer, loader):
    model.train()

    losses, iou_scores, f1_scores, f2_scores = [], [], [], []

    with tqdm(loader, desc="Training", unit="batch") as tbar:
        for idx, (img, mask) in enumerate(tbar):
            # move data and targets to device (gpu or cpu)
            img = img.float().to(DEVICE)
            mask = mask.to(DEVICE)
            with torch.cuda.amp.autocast():
                # making prediction
                pred = model(img)
                # calculate loss and dice coefficient and append it to losses and metrics
                Loss = mixed_dice_bce_loss(pred, mask)

                tp, fp, fn, tn = smp.metrics.get_stats(
                    torch.sigmoid(pred),
                    mask.to(torch.int64),
                    threshold=0.6,
                    mode="binary",
                    num_classes=1,
                )
                iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                f2_score = smp.metrics.fbeta_score(
                    tp, fp, fn, tn, beta=2, reduction="micro"
                )
                losses.append(Loss.item())
                iou_scores.append(iou_score)
                f1_scores.append(f1_score)
                f2_scores.append(f2_score)

                # Update tqdm description with current metrics for each batch
                tbar.set_postfix(
                    loss=f"{Loss.item():.4f}",
                    iou=f"{iou_score:.4f}",
                    f1=f"{f1_score:.4f}",
                    f2=f"{f2_score:.4f}",
                )

            # backward
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

        # Set final metrics after the loop finishes
        tbar.set_postfix(
            loss=f"{sum(losses) / len(losses):.4f}",
            iou=f"{sum(iou_scores) / len(iou_scores):.4f}",
            f1=f"{sum(f1_scores) / len(f1_scores):.4f}",
            f2=f"{sum(f2_scores) / len(f2_scores):.4f}",
        )


# 1 validation loop
def val_loop(model, loader):
    model.eval()

    with torch.no_grad():
        losses, iou_scores, f1_scores, f2_scores = (
            [],
            [],
            [],
            [],
        )
        with tqdm(loader, desc="Testing", unit="batch") as tbar:
            for idx, (img, mask) in enumerate(tbar):
                # move data and targets to device(gpu or cpu)
                img = img.float().to(DEVICE)
                mask = mask.to(DEVICE)

                # making prediction
                pred = model(img)

                # calculate loss and dice coefficient and append it to losses and metrics
                Loss = mixed_dice_bce_loss(pred, mask)

                tp, fp, fn, tn = smp.metrics.get_stats(
                    torch.sigmoid(pred),
                    mask.to(torch.int64),
                    threshold=0.6,
                    mode="binary",
                    num_classes=1,
                )
                iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                f2_score = smp.metrics.fbeta_score(
                    tp, fp, fn, tn, beta=2, reduction="micro"
                )

                losses.append(Loss.item())
                iou_scores.append(iou_score)
                f1_scores.append(f1_score)
                f2_scores.append(f2_score)

                # Update tqdm description with current metrics
                tbar.set_postfix(
                    loss=f"{sum(losses) / len(losses):.4f}",
                    iou=f"{sum(iou_scores) / len(iou_scores):.4f}",
                    f1=f"{sum(f1_scores) / len(f1_scores):.4f}",
                    f2=f"{sum(f2_scores) / len(f2_scores):.4f}",
                )
    return Loss


def train_augmentation(image_height, image_width):
    # augmentation for train
    TRANSFORM_TRAIN = A.Compose(
        [
            A.Resize(image_height, image_width),
            A.RandomBrightnessContrast(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            ToTensorV2(),
        ]
    )
    return TRANSFORM_TRAIN


# augmentation for test
TRANSFORM_VAL_TEST = A.Compose([ToTensorV2()])


def main(args: dict):
    """
    Parameters:
    - args (dict): dictionary of command-line arguments
    """
    # specify params from arg-parser
    epochs = args["epochs"] if args["epochs"] else EPOCHS
    lr = args["lr"] if args["lr"] else 3e-4
    batch_size_train = args["batch_size_train"] if args["batch_size_train"] else 16
    batch_size_val = args["batch_size_val"] if args["batch_size_val"] else 8
    images_path = args["images_path"] if args["images_path"] else ROOT_DIR_TRAIN
    masks_path = args["masks_path"] if args["masks_path"] else LABEL_PATH
    image_height = args["image_height"] if args["image_height"] else 512
    image_width = args["image_width"] if args["image_width"] else 512

    TRANSFORM_TRAIN = train_augmentation(image_height, image_width)

    # Define a collate function for the training dataset
    def train_collate_fn(batch):
        images, masks = zip(*batch)
        transformed_images = []
        transformed_masks = []
        for idx in range(len(images)):
            augmented = TRANSFORM_TRAIN(
                image=images[idx].numpy(), mask=masks[idx].permute(1, 2, 0).numpy()
            )
            transformed_images.append(augmented["image"])
            transformed_masks.append(augmented["mask"].permute(2, 0, 1))
        return torch.stack(transformed_images), torch.stack(transformed_masks)

    # Define a collate function for the validation and test datasets
    def val_test_collate_fn(batch):
        images, masks = zip(*batch)
        transformed_images = []
        transformed_masks = []
        for idx in range(len(images)):
            augmented = TRANSFORM_VAL_TEST(
                image=images[idx].numpy(), mask=masks[idx].permute(1, 2, 0).numpy()
            )
            transformed_images.append(augmented["image"])
            transformed_masks.append(augmented["mask"].permute(2, 0, 1))
        return torch.stack(transformed_images), torch.stack(transformed_masks)

    # if directory model exists than create this
    if not os.path.exists(SAVED_MODEL_PATH):
        os.makedirs(SAVED_MODEL_PATH)

    # define model and move it to device(gpu or cpu)
    model = smp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(DEVICE)

    # Create the ShipDataset instance
    ship_dataset = ShipDataset(root_dir=images_path, csv_file=masks_path)
    # Determine the size of the train, val, and test datasets based on the specified ratio.
    total_samples = len(ship_dataset)
    train_ratio = 0.9
    val_ratio = 0.1

    train_size = int(train_ratio * total_samples)
    val_size = math.ceil(val_ratio * total_samples)

    # Split the dataset into train, val, and test subsets using the torch.utils.data.random_split function.
    train_dataset, val_dataset = torch.utils.data.random_split(
        ship_dataset, [train_size, val_size]
    )

    # Create the DataLoader objects for each dataset to enable easy batching during training and evaluation.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=8,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        collate_fn=val_test_collate_fn,
        num_workers=8,
    )

    # checking whether the model needs to be retrained
    if LOAD_MODEL:
        print("Loading model from local state")
        model = smp.Unet(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )
        model.to(DEVICE)
        model.load_state_dict(torch.load(PATH_TO_MODEL))

    # Create optimizer only for trainable parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define the milestones and learning rates
    milestones = [3, 6]
    lr_values = [1e-4, 7e-5]

    # Create MultiStepLR scheduler
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.7)

    # Create ReduceLROnPlateau scheduler
    reduce_lr_scheduler = ReduceLROnPlateau(
        optimizer, factor=0.7, patience=3, min_lr=3e-5
    )
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")

        train_loop(model, optimizer, train_loader)
        scheduler.step()
        val_loss = val_loop(model, val_loader)
        reduce_lr_scheduler.step(val_loss)

        # save model
        torch.save(model.state_dict(), SAVED_MODEL_PATH + f"model{epoch + 1}.pth")


if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_height", type=int, help="Specify height for training")
    parser.add_argument("--image_width", type=int, help="Specify width for training")
    parser.add_argument(
        "--images_path", type=str, help="Specify path for train_v2 folder"
    )
    parser.add_argument(
        "--masks_path",
        type=str,
        help="Specify path for .csv file",
    )
    parser.add_argument(
        "--epochs", type=int, help="Specify number of epochs for model training"
    )
    parser.add_argument("--lr", type=float, help="Specify learning rate")
    parser.add_argument(
        "--batch_size_train", type=int, help="Specify batch size for training"
    )
    parser.add_argument(
        "--batch_size_val", type=int, help="Specify batch size for validation"
    )

    # Parsing command-line arguments
    args = parser.parse_args()
    args = vars(args)

    # Calling the main function with parsed arguments
    main(args)

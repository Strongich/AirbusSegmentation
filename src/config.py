import torch


SAVED_MODEL_PATH = "../models/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE_FULL = 768  # original dataset's images size
SIZE = 256  # size used for using in model (3x3 squares to make original image)
ROOT_DIR_TRAIN = "../data/train_v2"
ROOT_DIR_TEST = "../data/test_v2"
LABEL_PATH = "../data_cleared/uniqueAllLabels.csv"
LOAD_MODEL = False
OUTPUT_PATH = "../runs/"

"""

Author: Annam.ai IIT Ropar
Team Name: OverTakers
Team Members: Member-1 rishu Shukla , member-2 Kaushik pal
Leaderboard Rank: 55

"""import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import copy

# Paths
BASE_DIR = "/kaggle/input/soil-classification/soil_classification-2025"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
LABELS_CSV = os.path.join(BASE_DIR, "train_labels.csv")
TEST_IDS_CSV = os.path.join(BASE_DIR, "test_ids.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
.

def preprocessing():
    print("This is the file for preprocessing")
  return 0

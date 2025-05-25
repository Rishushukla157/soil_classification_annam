"""

Author: Annam.ai IIT Ropar
Team Name: OverTakers
Team Members: Member-1 rishu Shukla , member-2 Kaushik pal
Leaderboard Rank: 55

"""
import os
import pandas as pd
import numpy as np
import shutil
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def preprocessing():
    print("This is the file for preprocessing")
  return 0

# Installations
import os

os.system("pip install moviepy torch pandas numpy --quiet")
os.system("git clone https://github.com/ultralytics/yolov5.git")
os.listdir()
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import cv2
import shutil
import numpy as np
import pandas as pd
from time import sleep
import torch
import glob
from datetime import timedelta as delta
from moviepy.editor import VideoFileClip  # to get video duration
from utils import *


def detection(video_path):
    print("Detection Working!")


detection("Apple")

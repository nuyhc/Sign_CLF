import streamlit as st 
import os
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import koreanize_matplotlib
import seaborn as sns
from PIL import Image
import pillow_heif
import cv2
from skimage.io import imread
from skimage.transform import resize

# import splitfolders

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report

import glob
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Likelion AI School Sign Miniproject",
    page_icon="🐶",
    layout="wide",
)
st.sidebar.markdown("SIGN SIGN")

st.title("SIGN")

st.write("""
SIGN
""")
filename = st.file_uploader("Choose a file")

st.text(type(filename))

model = keras.models.load_model('model/model_kor_num_no_augmentation.h5')


def convert_letter(result):
    classLabels = {idx:c for idx, c in zip(idx, alpha)}
    try:
        res = int(result)
        return classLabels[res]
    except:
        return "err"


def img_resize_to_gray(filename):
    """파일 경로를 입력 받아 사이즈 조정과 그레이로 변환하는 함수

    Args:
        filename (str): 파일 경로
    Returns:
        arr (np.array)
    """
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (300, 300))
    return img

# def upload_and_predict(filename):
#     # img = Image.open(filename)
#     img = cv2.imread(filename)
#     # img = cv2.imread(filename)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.resize(img, (300, 300))
#     # img = Image.open(filename)
#     # img = img.convert('RGB')
#     # img = img.resize((300, 300))
#     plt.figure(figsize=(4, 4))
#     plt.imshow(img)
#     plt.axis('off')


#     # img = Image.open(filename)
#     # img = img.convert('RGB')
#     # img = img.resize((300, 300))
#     # # show image
#     # plt.figure(figsize=(4, 4))
#     # plt.imshow(img)
#     # plt.axis('off')
#     # # predict
#     # # img = imread(filename)
#     # # img = preprocess_input(img)
#     probs = model.predict(np.expand_dims(img, axis=0))
#     return convert_letter(np.argmax(model.predict(img)))

# pred = np.argmax(model.predict(img_resize_to_gray(filename).reshape(1, 300, 300, 1)))


if filename is not None:
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(filename, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(filename, (300, 300))    
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')



    # img = imread(filename)
    # img = preprocess_input(img)
    pred = np.argmax(model.predict(img.reshape(1, 300, 300, 1)))
    # text = []
    st.image(img, use_column_width=False)
    st.text(pred)
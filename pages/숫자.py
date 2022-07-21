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

# import splitfolders

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

model = keras.models.load_model('model/model_kor_num_no_augmentation.h5')


alpha = [chr(x).upper() for x in range(97, 123)]
alpha.remove("J")
alpha.remove("Z")
idx = [x for x in range(0, 24)]


def convert_letter(result):
    classLabels = {idx:c for idx, c in zip(idx, alpha)}
    try:
        res = int(result)
        return classLabels[res]
    except:
        return "err"


# def upload_and_predict2(filename):
#     img = Image.open(filename)
#     img = img.convert('RGB')
#     img = img.resize((224, 224))
#     print(img.size)
#     # show image
#     plt.figure(figsize=(4, 4))
#     plt.imshow(img)
#     plt.axis('off')
#     # predict
# #     img = imread(filename)
# #     img = preprocess_input(img)
#     probs = pretrained_model.predict(np.expand_dims(img, axis=0))
#     for idx in probs.argsort()[0][::-1][:8]:
#         print("{:.2f}%".format(probs[0][idx]*100), "\t", label_maps_rev[idx].split("-")[-1])

def upload_and_predict2(filename):
    img = Image.open(filename)
    img = img.convert('RGB')
    img = img.resize((300, 300))
    print(img.size)
    # show image
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    # predict
#     img = imread(filename)
#     img = preprocess_input(img)
    probs = model.predict(np.expand_dims(img, axis=0))
    return convert_letter(np.argmax(model.predict(img.reshape(1, 28, 28, 1))))
   


if filename is not None:
    img = Image.open(filename)
    img = img.convert('RGB')
    img = img.resize((300, 300))
    
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')

    probs = model.predict(np.expand_dims(img, axis=0))
    # text = []
    st.image(img, use_column_width=False)
    st.text(convert_letter(np.argmax(model.predict(img))))
import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from PIL import Image





@st.cache
def fn_load():   
    std_transformer = load('./std_transformer.joblib')  
    pca_transformer = load('./pca_transformer.joblib')    
    clf = load('./img_model_v3.joblib')  
    return std_transformer, pca_transformer, clf


def fn_predict(img, std_transformer, pca_transformer, clf):    
    img_array = np.array(img).flatten()
    std_img = std_transformer.transform(img_array.reshape(1, -1))
    pca_img = pca_transformer.transform(std_img)
    label = clf.predict(pca_img)
    return f'species = {label}'


st.title('BIRD CLASSIFICATION APP')
img_ = st.file_uploader("CHOOSE JPG IMAGE FILE", type='jpg')


if img_ is not None:
    img = Image.open(img_).resize((112, 112))
    std_transformer, pca_transformer, clf = fn_load()
    y_pred = fn_predict(img, std_transformer, pca_transformer, clf)
    st.write(y_pred)

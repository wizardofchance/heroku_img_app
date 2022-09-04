import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
from PIL import Image




st.title('BIRD CLASSIFICATION APP')

df_pca_ = st.file_uploader("CHOOSE CSV FILE WITH PCA COMPONENTS")
df_mean_std_ = st.file_uploader("CHOOSE CSV FILE WITH IMG MEAN & STD OF EACH FEATURE")
clf_ = st.file_uploader("CHOOSE CLASSIFIER JOBLIB FILE")
img_ = st.file_uploader("CHOOSE JPG IMAGE FILE", type='jpg')



@st.cache
def fn_load():      
    df_pca = pd.read_csv(df_pca_)
    df_mean_std = pd.read_csv(df_mean_std_)
    clf = load(clf_)  
    return df_pca, df_mean_std, clf


def fn_predict(df_pca, df_mean_std, clf, img):
    
    img_array = np.array(img).flatten()
    
    means = df_mean_std.means.values
    stds = df_mean_std.stds.values
    std_img = (img_array - means)/stds

    pca_img = std_img @ df_pca.values.T 
    label = clf.predict(pca_img.reshape(1, -1))

    return f'species = {label}'

c1 = df_pca_ is not None  
c2 = df_mean_std_ is not None  
c3 = clf_ is not None
c4 = img_ is not None
all_files_loaded = c1 and c2 and c3 and c4

if all_files_loaded:
    img = Image.open(img_).resize((112, 112))
    df_pca, df_mean_std, clf = fn_load()
    st.write(fn_predict(df_pca, df_mean_std, clf, img))

  
        
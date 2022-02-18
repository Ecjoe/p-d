#type: ignore
from distutils.command.upload import upload
import streamlit as st
from PIL import Image 
import pandas as pd
import numpy as np
import tensorflow as tf
import os
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Pneumonia Diagnosis')

@st.cache(allow_output_mutation=True)
def loading_model():
    model_path = "models\ensemble_model.h5"
    model = tf.keras.models.load_model(model_path)
    return model
model = loading_model()

uploaded_file = st.file_uploader("Upload your X-ray Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # file_details = {"filename":uploaded_file.name, "filetype":uploaded_file.type, "filesize":uploaded_file.size}
    # st.write(file_details)

    image = Image.open(uploaded_file)
    st.image(image, width = 250)

    saved_file_path = os.path.join('saved', uploaded_file.name)
    with open(saved_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success('File Saved!')

    # print(uploaded_file.name)
    # saved_path = os.path.join('sa')
    img = tf.keras.preprocessing.image.load_img(saved_file_path, target_size = (224, 224), )
    img = tf.keras.preprocessing.image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    # img = img / 255.0 

    # print(x.shape)
    # print(x)



    preds = model.predict(img).reshape(1,-1)[0]
    # # y = preds.astype("int32")

    print(preds)
    # print(preds.shape)

    # y = preds
    # print(y)
    # print(y.shape)

    if preds >=0.5:
        st.subheader('Result : Pneumonia')
    else:
        # st.balloons()
        st.subheader('Result : Normal')


# st.write('cwbiq')



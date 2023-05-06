# Import all the dependencies
import streamlit as st
import os 
import imageio
import sys
import numpy as np

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model
from tensorflow import keras
from tensorflow.keras import layers

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Set up the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('Lip Reading')
    st.info('This application is developed from the LipNet deep learning model.')


st.title('LipNet Full Stack App') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('.', 'data', 's1'))
print(options)


selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
        #os.system(f'ffmpeg -i {file_path} test_video.gif -y')

    # Rendering the app
        #video = open('test_video.mp4', 'rb') 
        #video_bytes = video.read() 
       
        #video_bytes = tf.io.read_file('test_video.mp4')
        
       
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        video_tensor = tf.contrib.ffmpeg.decode_video(video_bytes)
        #video_tensor = tf.io.decode_video(video_bytes)
        #video_tensor = tf.io.decode_gif(video_bytes)
        st.video(video_bytes)

    """
        vid = imageio.get_reader('test_video.mp4', 'ffmpeg')
        data = np.stack(list(vid.iter_data()))

        img_bytes = [tf.image.encode_jpeg(frame, format='rgb') for frame in data]
    with tf.Session() as sess: 
        img_bytes = sess.run(img_bytes)
    """

    with col2: 
        """
        st.info('This is all the machine learning model sees when making a prediction')
        #print("bbbbbb", file_path)
        string_tensor_filepath = tf.constant(selected_video)
        video, annotations = load_data(tf.convert_to_tensor(string_tensor_filepath))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400) 
        """

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        #yhat = model.predict(tf.expand_dims(video, axis=0))
        yhat = model.predict(tf.expand_dims(video_tensor, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)

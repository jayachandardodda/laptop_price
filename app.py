import streamlit as st
import pickle
import  numpy as np
import math

# import model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title('Laptop Price Predictor')

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
Type = st.selectbox('Type', df['TypeName'].unique())

# ram
RAM = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight
Weight = st.number_input('Weight of the laptop')

# Touchscreen
touchscreen = st.selectbox('Tochscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
# cpu
cpu = st.selectbox('Cpu', df['cpu_brand'].unique())

# HDD
HDD = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
SSD = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
# Gpu_brand
gpu = st.selectbox('GPU', df['Gpu_brand'].unique())

OS = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
     ppi = None
     if touchscreen == 'Yes':
          touchscreen = 1
     else:
          touchscreen = 0
     if ips == 'Yes':
          ips = 1
     else:
          ips = 0

     x_res = int(resolution.split('x')[0])
     y_res = int(resolution.split('x')[1])
     ppi = ((x_res) ** 2 + (y_res) ** 2) ** 0.5 / screen_size


     query = np.array([company,Type,RAM,Weight,touchscreen,ips,ppi,cpu,HDD,SSD,gpu,OS])
     query = query.reshape(1,12)
     st.title(np.exp(pipe.predict(query)))
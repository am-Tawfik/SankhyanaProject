# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



# Importing pandas and numpy to deal with data
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import requests

# load the .sav model from github
url = 'https://github.com/am-Tawfik/SankhyanaProject/blob/main/trained_logistic_model.sav'

# download the file
model = requests.get(url)

# save the downloaded model to a temporary file
with open("trained_logistic_model", "wb") as f:
    pickle.dump(model, f)

# laod the saved model
with open("trained_logistic_model", "rb") as f:
    model = pickle.load(f)
    
    
    
def BreastCancerPredict (new_data):
    
    # creating an array
    new_data_array = np.array(new_data)
    
    # reshaping the array for predicting
    # and make it as a row
    new_data_reshaped = new_data_array.reshape(1,-1)
    
    # use the loaded model to predict the new observation
    prediction = model.predict(new_data_reshaped)
    
    # Classify the new prediction
    if prediction[0] == 0:
        return "The person doesn't has Breast Cancer"
    else:
        return "The person has Breast Cancer"


def main():
    
    # interface title
    st.title("Breast Cancer Prediction")
    
    # getting input from the user
    mean_radius = st.text_input("Enter the mean radius - range:(1,30)")
    mean_texture = st.text_input("Enter the mean texture- range:(1,40)")
    mean_perimeter = st.text_input("Enter the mean perimeter- range:(40,200)")
    mean_area = st.text_input("Enter the mean area- range:(140,2550)")
    mean_smoothness = st.text_input("Enter the mean smoothness- range:(0.01,0.2)")
    
    # transform to numeric and convert the non float input to NaN
    mean_radius = pd.to_numeric(mean_radius, errors = 'coerce')
    mean_texture = pd.to_numeric(mean_texture, errors = 'coerce')
    mean_perimeter = pd.to_numeric(mean_perimeter, errors = 'coerce')
    mean_area = pd.to_numeric(mean_area, errors = 'coerce')
    mean_smoothness = pd.to_numeric(mean_smoothness, errors = 'coerce')
    
    # Take the log
    mean_radius = np.log(mean_radius)
    mean_texture = np.log(mean_texture)
    mean_perimeter = np.log(mean_perimeter)
    mean_area = np.log(mean_area)
    mean_smoothness = np.log(mean_smoothness)
    
    # Normalize the values
    mean_radius = (mean_radius - 6.981) / (28.11-6.981)
    mean_texture = (mean_texture - 9.71) / (39.28-9.71)
    mean_perimeter = (mean_perimeter - 43.79) / (188.5-43.79)
    mean_area = (mean_area - 143.5) / (2501-143.5)
    mean_smoothness = (mean_smoothness - 0.05263) / (0.1634-0.05263)

    # list of independent variables
    new_observation = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]
    
    # Diagnosis
    diagnosis = ''
    
    if st.button('Predict'):
        diagnosis = BreastCancerPredict(new_observation)
        st.success(diagnosis)

if __name__ =='__main__':
    main()
    
    
    
    
    
    
    
    

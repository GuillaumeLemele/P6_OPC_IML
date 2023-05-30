# -*- coding: utf-8 -*-
"""
Created on Thu May 25 00:33:24 2023

@author: Lemel
"""
# cd C:\Users\Lemel\OPC-P6
# python programe_final_OPC6.py

# Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import json
import os

# Define constants for resizing pictures
img_height = 180
img_width = 180

# Define paths (With folder inside 15 jpeg random pictures on Google Image Research for 15 breeds of dog)
model_path = 'C:\\Users\\Lemel\\OPC-P6\\my_model.h5'
class_indices_path = 'C:\\Users\\Lemel\\OPC-P6\\class_indices.json'
img_folder = 'C:\\Users\\Lemel\\OPC-P6\\predictpictures'

# Load our trained model made with the pretrained Xception model 
my_xcept_model = load_model(model_path)

# Load the class indices
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

def predict_breed(img_path, model):
    # Load the image file, resizing it to 180x180 pixels (as required by the model)
    img = image.load_img(img_path, target_size=(img_height, img_width))
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Normalize the image array (rescale pixel values from [0, 255] to [0, 1])
    img_array /= 255.
    # Add an extra dimension for batch size (model expects a batch)
    img_batch = np.expand_dims(img_array, axis=0)
    # Get the predicted probabilities for each class
    predictions = model.predict(img_batch)
    # Get the index of the highest probability
    breed_index = np.argmax(predictions)
    # Get the name of the breed (The class indices in our model match the order in the class_indices dictionary)
    breed = list(class_indices.keys())[breed_index]
    return breed

# Iterate over images in the img_folder
for img_file in os.listdir(img_folder):
    img_path = os.path.join(img_folder, img_file)

    # Predict breed
    predicted_breed = predict_breed(img_path, my_xcept_model)
    
    # Remove prefix from predicted breed
    predicted_breed = predicted_breed.split('-')[-1]

    # Get the breed from the image file name as we labelled them during our Google research (without extension)
    true_breed = os.path.splitext(img_file)[0]
    
    # Display the result
    print(f'True Breed: {true_breed}\nPredicted Breed: {predicted_breed}\n---')

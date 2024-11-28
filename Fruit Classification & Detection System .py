# --------------------------------------------------------------------------------------------------------------------------
# Code created by: Mohammed Safwanul Islam @safwandotcom®
# Project: Fruit Classification & Detection System Data Science Image Processing using Data Sets 
# Date created: 15th November 2024
# Organization: N/A
# --------------------------------------------------------------------------------------------------------------------------
# Description:
#   The code continuously captures video frames from the webcam.
#   For each frame, it resizes and preprocesses the image, then feeds it to a trained model for classification.
#   The predicted class label is overlayed on the video frame.
#   The video is displayed with the prediction label, and the loop continues until the user presses the 'q' key to exit.
#   When the loop exits, resources are released and all OpenCV windows are closed.
#   This code can be used for real-time image classification or object detection in a video stream, 
#   such as identifying fruits, animals, or any other classes that your model is trained on.
# --------------------------------------------------------------------------------------------------------------------------
# License:
# This code belongs to @safwandotcom®.
# Code can be freely used for any purpose with proper attribution.
# --------------------------------------------------------------------------------------------------------------------------

# Modules to install for this program to run using WINDOWS POWERSHELL
# python3 -m pip install tensorflow[and-cuda] (tensor flow GPU version) or if failed to install can use 
# pip install numpy 
# pip install tensorflow
# pip install opencv-python
# pip install mediapipe
# pip install matplotlib
# --------------------------------------------------------------------------------------------------------------------------


import os #For interacting with the operating system (e.g., defining file paths).
import numpy as np  #For numerical computations and array manipulation.
import matplotlib.pyplot as plt #For creating plots (commented out in this code).
import tensorflow as tf #The core TensorFlow library for building and training machine learning models.
import cv2 # OpenCV for computer vision tasks like image processing and video capture.

#This below line imports the Sequential class from TensorFlow's Keras API. The Sequential model is a linear stack of layers, meaning you can just add one layer after another.
from tensorflow.keras.models import Sequential

#This line imports several layers from the Keras API:
#Dense: A fully connected (dense) layer where every neuron is connected to every neuron in the previous layer. It's commonly used for the final layers of a neural network for classification or regression tasks.
#Flatten: This layer flattens the input data (usually from a multi-dimensional format, like a 2D image, into a 1D array). It's often used before feeding the data into dense layers.
#Dropout: A layer used to prevent overfitting during training by randomly setting a fraction of the input units to zero during each forward pass. It helps regularize the model by reducing dependency on particular neurons.
from tensorflow.keras.layers import Dense, Flatten, Dropout

# MobileNetV2 is a pre-trained convolutional neural network (CNN) architecture, 
# optimized for mobile and edge devices. It has been pre-trained on the ImageNet dataset and 
# can be used either as a feature extractor or as a fine-tuning model for transfer learning.
from tensorflow.keras.applications import MobileNetV2

#This below line imports several utilities for image preprocessing:

# ImageDataGenerator: A class for real-time data augmentation: 
# It allows to perform operations like rotation, zoom, flipping, or rescaling on images during training. 
# This helps improve model generalization and prevents overfitting by providing the model with varied versions of the same images.
#img_to_array: A function to convert a PIL image into a NumPy array. It's often used to prepare an image for feeding into a neural network.
#load_img: #A function that loads an image from a file, converting it into a PIL image object, which can then be further processed (e.g., resized, converted to an array).
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, # Normalize pixel values to [0, 1]
    rotation_range=30, #Randomly rotate images by upto 20 degrees
    width_shift_range=0.2, # Randomly shift images horizontally by up to 20%
    height_shift_range=0.2, # Randomly shift images vertically by up to 20%
    shear_range=0.2, # Randomly apply shear transformation
    zoom_range=0.2, # Randomly zoom images by up to 20%
    horizontal_flip=True, ## Randomly flip images horizontally
    fill_mode='nearest' # Fill in pixels after transformations using nearest pixel
)

# Only rescaling for test data 
test_datagen = ImageDataGenerator(rescale = 1.0 / 255)

# Load training data using ImageDataGenerator
train_data = train_datagen.flow_from_directory(r"C:\Users\HP\Downloads\train", #Path of my training data from my PC
    target_size=(224, 224), # Resize images to model's input size
    batch_size=32, # Number of images per batch
    class_mode='categorical', # One-hot encode labels (multi-class classification)
    color_mode='rgb' # Assume RGB images)
)

# Load test data
test_data = test_datagen.flow_from_directory(r"C:\Users\HP\Downloads\test", # Replace with test data path from PC
    target_size=(224, 224), #This specifies the desired size of the input images for the model. All images will be resized to this dimension.
    batch_size=32, #This determines the number of images to be processed in each batch during training or testing.
    class_mode='categorical', #This indicates that the model is performing multi-class classification. The labels will be one-hot encoded, where each class is represented by a binary vector.
    color_mode='rgb' #This specifies that the images are in RGB color format.
)

# Load the MobileNetV2 model with ImageNet weights, excluding the top layer
# #This line loads the pre-trained MobileNetV2 model with the following parameters:

#weights='imagenet': This specifies that the model should load weights that were pre-trained on the ImageNet dataset. 
#ImageNet is a large dataset with millions of images across 1000 categories, so these weights are a good starting point for most image classification tasks.
#include_top=False: This means that the fully connected (top) layers of the model (the layers used for final classification) will not be included. 
#input_shape=(224, 224, 3): This specifies the input size of the images that the model expects. 
#224x224 is a common size for models like MobileNetV2. 3 indicates that the images are in RGB format (3 channels).
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) 
base_model.trainable = False  #Freeze the base layers means that no gradients will be computed for the layers of the base model, and the weights will remain fixed. 
#This is important in transfer learning because we usually want to keep the pre-trained features intact (since they have already learned useful patterns) and only train the additional layers you add on top.

# Build the model
model = Sequential([
    base_model, #By freezing the base model's layers (as done earlier), we leverage these learned features without training them again, saving computational resources and potentially improving performance.
    Flatten(), #This layer flattens the output of the previous layer (the feature maps from MobileNetV2) into a one-dimensional array.
    Dropout(0.5), #This layer randomly drops 25% of the neurons during training, helps prevent overfitting by reducing the model's reliance on specific neurons.
    Dense(128, activation='relu'), #This is a densely connected layer with 128 neuron, The relu activation function introduces non-linearity to the model, allowing it to learn complex patterns.
    Dense(10, activation='softmax')  # The softmax activation function ensures that the output probabilities for each class sum to 1, providing a probability distribution over the 10 classes.
])

#MODEL COMPILATION

#model.compile(...): Compiles the model, specifying the optimizer, loss function, and metrics to track during training.
#optimizer='adam': Adam is a popular optimizer often used for its efficiency and effectiveness in finding good parameter values.
# loss='categorical_crossentropy': Since we're dealing with multi-class classification (10 fruit classes), 
# categorical_crossentropy is a suitable loss function that measures the error between the predicted class probabilities and the true labels (one-hot encoded). 
# metrics=['accuracy']: We'll track accuracy as the metric to measure the model's performance during training and validation.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit( #now it trains the model using the compiled configuration. 
    train_data, #The ImageDataGenerator object providing batches of training images, labels, and augmentations.
    epochs=15, #Number of training iterations (adjust based on your dataset and performance).
    validation_data=test_data, #The ImageDataGenerator object for validation data, allowing the model to monitor its performance against unseen data.
)

# Plot training and validation accuracy
plt.figure(figsize=(5, 5))  #Create a new figure with a size of 5x5 inches
plt.plot(history.history['accuracy'], color='red', label='Training Accuracy') # Plot training accuracy (in red)
plt.plot(history.history['val_accuracy'], color='blue', label='Validation Accuracy') # Plot validation accuracy (in blue)
plt.xlabel('Epochs') # Label for the x-axis
plt.ylabel('Accuracy') # Label for the y-axis
plt.title('Training and Validation Accuracy') # Title for the plot
plt.legend() # Show the legend to distinguish between training and validation accuracy
plt.show() # Display the plot

#Plot training and validation loss
plt.figure(figsize=(5, 5)) # Create a new figure with a size of 5x5 inches
plt.plot(history.history['loss'], color='red', label='Training Loss') # Plot training loss (in red)
plt.plot(history.history['val_loss'], color='blue', label='Validation Loss') # Plot validation loss (in blue)
plt.xlabel('Epochs') # Label for the x-axis
plt.ylabel('Loss') # Label for the y-axis
plt.title('Training and Validation Loss') # Title for the plot
plt.legend() # Show the legend to distinguish between training and validation loss
plt.show() # Display the plot

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the trained model
model.save('fruit_classifier_model.h5')

# Define fruit labels (make sure these match the class indices)
labels = ['Apple', 'Banana', 'Durian', 'Jackfruit', 'Mango', 
          'Orange', 'Pineapple', 'Pomegranate', 'Tomato', 'Watermelon']

# Open the webcam & Initialize video capture from the default camera (0), if want to use secondary camera can put 1 instead of zero and respectively onwards
cap = cv2.VideoCapture(0)
#You can also put video from your device instead of 0 by replacing with /r"YOUR PATH FROM DIRECTORY"

while True: #This creates an infinite loop that continuously captures frames from the video stream until the user decides to stop the loop (using the key 'x').
    # Capture a frame from the video stream

    #  ret: A boolean indicating whether the frame was successfully captured.
    #frame: The actual image frame captured from the video.
    ret, frame = cap.read() #cap.read(): This reads the next frame from the video stream (usually from a webcam). It returns two values:
    if not ret: #If ret is False, this means the frame could not be captured (e.g., video stream has ended or there was an error). The loop will break if this happens.
        break

    # Preprocess the frame for model input
    img = cv2.resize(frame, (224, 224))  # Resize to model's input size

    # Normalize & converts the resized image into a NumPy array, which is the format that Keras models accept for prediction.
    #/ 255.0: Normalizes the image data by dividing each pixel value by 255.0 to scale the pixel values to the range [0, 1] (as required by most pre-trained models).
    img = img_to_array(img) / 255.0  

    img = np.expand_dims(img, axis=0)  # Add batch dimension & The model expects a batch of images, but we only have one image here. 
    #This line adds a batch dimension (converting the shape from (224, 224, 3) to (1, 224, 224, 3)).

    # Predict the class
    predictions = model.predict(img) # This makes a prediction using the trained model. It outputs the predicted class probabilities for the input image.
    class_index = np.argmax(predictions) #This finds the index of the class with the highest probability.
    label = labels[class_index] #This retrieves the corresponding class label (e.g., the name of the fruit) from the labels list or array based on the predicted class index.

    # Display the prediction on the frame
    #cv2.putText(): This function adds text to the image (the video frame). 
    #frame: The image frame to which the text will be added.
    #f"It matches with: {label}": The text to display, which will be the predicted class label
    #(10, 30): The position on the frame where the text will start (coordinates in pixels).
    #cv2.FONT_HERSHEY_SIMPLEX: The font type for the text.
    #1: The font scale (size of the text).
    #(0, 255, 0): The color of the text (green in this case, as OpenCV uses BGR format).
    #2: The thickness of the text.
    cv2.putText(frame, f"It matches with: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Show the frame
    #cv2.imshow(): This function displays the current frame in a window.
    #"Fruit Detector by Safwan & Arsyi": The title of the window that will display the video.
    #frame: The current video frame with the prediction text displayed on it.
    cv2.imshow("Fruit Detector by Safwan & Arsyi", frame)

    # Break loop on 'q' key press
    #cv2.waitKey(1): This waits for a key press. The 1 argument means it will check for key presses every 1 millisecond.
    #& 0xFF == ord('x'): If the 'q' key is pressed, the loop breaks. The ord('q') gives the ASCII value of the 'q' key. 
    # The 0xFF bitwise operation ensures compatibility with different operating systems.
    #
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release() #cap.release(): Releases the video capture object. This is important to free up the resources used by the video capture device (e.g., webcam).
cv2.destroyAllWindows() #cv2.destroyAllWindows(): Closes all OpenCV windows, including the video display window.




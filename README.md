# Hand_Written_Recognition_for_Children_Using_Streamlite

## Introduction
This project is designed to help Children (about 5th std) understand how computers can recognize handwritten digits. We'll use a special type of software called a neural network to do this. Children can draw a digit on a canvas and they can change the parameters of the model for fun  , and the computer will try to recognize it!

## Prerequisites
Before you start, make sure you have the following Python libraries installed:
- `PIL` (Pillow): For image processing.
- `streamlit`: For creating the interactive web application.
- `torch`: For deep learning and neural networks.
- `scikit-learn`: For machine learning operations, such as train-test splitting and calculating accuracy.
- `numpy`: For numerical operations.
- `opencv-python`: For image processing.
- `skorch`: For integrating PyTorch with scikit-learn.
- `matplotlib`: For data visualization.
- `gtts`: For converting text to speech.
- `IPython`: For displaying audio.
- `os`: For file and directory operations.

You can install these libraries using the `pip install` command.

## Project Components

### 1. Data Preparation
- The project starts by fetching the MNIST dataset, which contains handwritten digits and their labels.
- The data is preprocessed by scaling it to values between 0 and 1.

### 2. Model Selection and Training
- You can choose between two models: "MLP Classifier" and "Convolutional Neural Network."
- The neural network models are defined using PyTorch. The "MLP Classifier" consists of two linear layers with dropout, and the "Convolutional Neural Network" includes convolutional and fully connected layers.
- You can customize model parameters like Dropout Rate, Learning Rate, and Max Epochs to control how the model learns.
- Training is performed using the selected model and parameters.

### 3. Visualization and Prediction
- You can view example images of handwritten digits.
- You can draw a digit on the canvas using the web application.
- After drawing, you can click the "Recognize Digit" button to make a prediction.
- The predicted digit, along with model parameters and accuracy, is displayed.
- An audio file is generated to announce the predicted digit, and you can listen to it by clicking the play button.

### 4. Learning More
- If you want to learn more about how the computer recognizes digits, click the "Learn More" button.

## How It Works
- When you draw a digit, the computer uses a neural network to compare it to examples it has seen before.
- The neural network tries to make a guess based on what it has learned from those examples.
- You get to change the settings (Dropout Rate, Learning Rate, Max Epochs) to see how they affect the computer's learning and prediction.

## Conclusion
This project is a fun and educational way to understand how computers can recognize handwritten digits using neural networks especially designed for children about 5the class .They are free to explore, draw different digits, and see how changing the model parameters impacts the results!

Have fun and happy learning!
# Handwritten Digit Recognition Using Streamlit

![Starting Page](https://github.com/KALANITHII/Hand_Written_Recognition_for_Children_Using_Streamlite/assets/94750661/4cd5dacd-7dba-4509-8dbe-c96387934f6f)
![Drawing Page](https://github.com/KALANITHII/Hand_Written_Recognition_for_Children_Using_Streamlite/assets/94750661/0a44d805-8af5-47e7-ad01-9a916b52f190)
![Prediction Page](https://github.com/KALANITHII/Hand_Written_Recognition_for_Children_Using_Streamlite/assets/94750661/d3dafc39-2d1d-46f5-ad5b-2af7f0e42767)



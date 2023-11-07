
from PIL import Image
import random
import streamlit as st
import torch
from sklearn.metrics import accuracy_score
from streamlit_drawable_canvas import st_canvas
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from  skorch import NeuralNetClassifier
from io import BytesIO
import matplotlib.pyplot as plt
from gtts import gTTS
import IPython.display as ipd
import os

mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')
X /= 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
XCnn = X.reshape(-1, 1, 28, 28)
XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=42)



def get_accuracy(model_name, dropout, learning_rate, max_epochs):
    if model_name == "MLP Classifier":
        class ClassifierModule(nn.Module):
            def __init__(self):
                super(ClassifierModule, self).__init__()
                self.dropout = nn.Dropout(dropout)
                self.hidden = nn.Linear(784, 100)
                self.output = nn.Linear(100, 10)

            def forward(self, X, **kwargs):
                X = F.relu(self.hidden(X))
                X = self.dropout(X)
                X = F.softmax(self.output(X), dim=-1)
                return X

        net = NeuralNetClassifier(
            ClassifierModule,
            max_epochs=max_epochs,
            lr=learning_rate,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        net.fit(X_train, y_train)
        y_pred = net.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return net, accuracy

    elif model_name == "Convolutional Neural Network":
        class Cnn(nn.Module):
            def __init__(self):
                super(Cnn, self).__init()
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
                self.conv2_drop = nn.Dropout2d(p=dropout)
                self.fc1 = nn.Linear(1600, 100)
                self.fc2 = nn.Linear(100, 10)
                self.fc1_drop = nn.Dropout(p=dropout)

            def forward(self, x):
                x = torch.relu(F.max_pool2d(self.conv1(x), 2))
                x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
                x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
                x = torch.relu(self.fc1_drop(self.fc1(x)))
                x = torch.softmax(self.fc2(x), dim=-1)
                return x

        cnn = NeuralNetClassifier(
            Cnn,
            max_epochs=max_epochs,
            lr=learning_rate,
            optimizer=torch.optim.Adam,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        cnn.fit(XCnn_train, y_train)
        y_pred_cnn = cnn.predict(XCnn_test)
        accuracy = accuracy_score(y_test, y_pred_cnn)
        return cnn, accuracy

def fig_to_image(fig):
    """Convert a Matplotlib figure to a PNG image."""
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

def plot_example(X, y):
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    for i in range(10):
        for j in range(10):
            index = i * 10 + j
            axes[i, j].imshow(X[index].reshape(28, 28), cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(y[index], fontsize=8)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.tight_layout()
    return fig

def recognize_digit():
    input_image = canvas_result.image_data
    if np.any(input_image):
        input_image = cv2.resize(input_image, (28, 28))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        input_image = input_image.reshape(1, -1)
        input_image = input_image.astype('float32')

        model, accuracy = get_accuracy(model_name, dropout, learning_rate, max_epochs)
        if model_name == "MLP Classifier":
            prediction = model.predict(input_image)
        elif model_name == "Convolutional Neural Network":
            input_image_cnn = input_image.reshape(-1, 1, 28, 28)
            prediction = model.predict(input_image_cnn)

        st.write(f"Model: {model_name}")
        st.write(f"Dropout Rate: {dropout}")
        st.write("Dropout Rate is like the magic trick that helps our model learn better. It randomly drops out or ignores some of the things it sees, which keeps it from getting too distracted. This way, our model can learn more efficiently!")
        st.write(f"Learning Rate: {learning_rate}")
        st.write("Learning Rate is like the speed of a car. If the learning rate is too high, the car might crash (learn too fast), and if it's too low, the car might never reach its destination (learn too slow). We need to find the right speed to teach our model.")
        st.write(f"Max Epochs: {max_epochs}")
        st.write("Max Epochs is like how many times we practice something. If we practice too many times, we might memorize everything (overfitting). If we don't practice enough, we might not be good (underfitting). We need to find the right number of practices to become experts!.")
        st.markdown(f"<p style='font-size: 24px; font-weight: bold; color: #ff5733; text-shadow: 2px 2px 4px #000000;'>Predicted Digit: {prediction[0]}</p>", unsafe_allow_html=True)
        display_chocolates(prediction[0])

        st.write(f"Accuracy: {accuracy:.2f}")

        tts = gTTS(f"The predicted digit is {prediction[0]}")
        tts.save("prediction.mp3")

        audio_file = open("prediction.mp3", "rb").read()
        st.audio(audio_file, format="audio/mp3")

        st.write("Click the play button to listen to the prediction.")
        ipd.Audio("prediction.mp3")

        os.remove("prediction.mp3")

def display_chocolates(predicted_digit):
    chocolate_image = Image.open('chocolate_image.png')
    
    st.write(f"Displaying {predicted_digit} chocolate(s):")
    for _ in range(predicted_digit):
        st.image(chocolate_image, use_column_width=True)


character_img = Image.open('character.png')

st.image(character_img, use_column_width=True)
st.title("Fun Handwritten Digit Recognition")

if st.button("How does this work?"):
    st.write("Hi there! I'm your friendly guide, Neural Nets. I'm here to help you learn about handwritten digits. You can draw a digit in the canvas, and I'll try to recognize it! Let's give it a try!")

st.sidebar.title("Model Selection and Parameters")
model_name = st.sidebar.selectbox("Select Model", ["MLP Classifier", "Convolutional Neural Network"])
dropout = st.sidebar.slider("Dropout Rate", 0.0, 1.0, 0.5,help="Dropout Rate is like the magic trick that helps our model learn better. It randomly drops out or ignores some of the things it sees, which keeps it from getting too distracted. This way, our model can learn more efficiently!")

if model_name == "MLP Classifier":
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1,help="Learning Rate is like the speed of a car. If the learning rate is too high, the car might crash (learn too fast), and if it's too low, the car might never reach its destination (learn too slow). We need to find the right speed to teach our model.")
    max_epochs = st.sidebar.slider("Max Epochs", 1, 100, 20, help = "Max Epochs is like how many times we practice something. If we practice too many times, we might memorize everything (overfitting). If we don't practice enough, we might not be good (underfitting). We need to find the right number of practices to become experts!.")
elif model_name == "Convolutional Neural Network":
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.002, help = "Learning Rate is like the speed of a car. If the learning rate is too high, the car might crash (learn too fast), and if it's too low, the car might never reach its destination (learn too slow). We need to find the right speed to teach our model.")
    max_epochs = st.sidebar.slider("Max Epochs", 1, 100, 10, help = "Max Epochs is like how many times we practice something. If we practice too many times, we might memorize everything (overfitting). If we don't practice enough, we might not be good (underfitting). We need to find the right number of practices to become experts!")

st.markdown("### Step 1: Draw a digit")
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=20,
    stroke_color="black",
    background_color="white",
    width=500,
    height=500,
    drawing_mode="freedraw",
)
if st.button("Recognize Digit"):
    if canvas_result.json_data:
        recognize_digit()

if st.button("Learn More"):
    st.write("Let me explain how I learn to recognize digits. I use a special brain called a neural network. It's like a super-smart robot that learns from examples. When you draw a digit, I compare it to the examples I've seen before and make a guess.")

example_images = plot_example(X_test[:100], y_test[:100])
st.image(fig_to_image(example_images), use_column_width=True, caption="Example Images")

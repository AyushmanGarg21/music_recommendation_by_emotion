# Music Recommendation System using Emotion Recognition

This project is a music recommendation system that utilizes emotion recognition to recommend songs based on the user's mood.

## Features
- Detects facial expressions to determine the user's emotion.
- Utilizes a Convolutional Neural Network (CNN) model for emotion recognition.
- Recommends music based on the detected emotion.

## Dataset
- FER2013 dataset for emotion recognition.
- OHAHEGA dataset for additional emotion data .
- music dataset for music recommendation (https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019).

## Requirements
- Python 3.x
- TensorFlow
- OpenCV
- streamlit
- Other necessary libraries (specified in requirements.txt)

## Downloads
- Download Modals form here - https://drive.google.com/drive/folders/1oznKpXVwOVVjMIfDdbyHoYvpK6AEFTT6?usp=sharing

## Usage
1. Install the required dependencies: `pip install -r requirements.txt`
2. Run the main in main_files script: `streamlit run app.py`
3. Follow the prompts to capture the user's facial expression.
4. Receive music recommendations based on the detected emotion.


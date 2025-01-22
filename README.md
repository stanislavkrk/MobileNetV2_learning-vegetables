# Description

MobileNetV2 learning to recognize vegetables in video flow.

dataset: https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset 

This project implements a system for recognizing vegetables in video streams and assigning them to clusters based on their similarities. It includes training a neural network, clustering features, and detecting objects in video.

# Files

- learning_MobileNetV2.py - MobileNetV2 neural network is trained on the dataset data, I add the file of weights as a separate file.
Created a vegetable dictionary for recognition (saved in a separate file).

- cluster_kmeans.py - based on the trained model, clustering of vegetables was performed using Kmeans (saved in a separate file).
  
- vegetable_model.h5 - trained model weights. You don`t need to train model again, use this file for recognizing.

- vegetable_clusters.json - a file with the results of clustering using the trained model.

- class_indices.json - dictionary of vegetables, wrote by model after training

- pytube_download.py - a script was created that downloads a video from YouTube for recognition. The script has several links to different vegetables, you can select any and download it. By default, broccoli is loaded. You can also switch to manual link input by the user.

- video_recognizer.py - a script for directly recognizing vegetables in a video stream.

- main.py - a file for selecting the above functions through a user-friendly interface.

# Features

Trains a MobileNetV2-based neural network to classify vegetables.
Supports data augmentation to improve robustness.

- Feature Extraction and Clustering:
  - Extracts features from trained images.
  - Uses K-Means clustering to group vegetables into clusters based on feature similarity.

- Real-Time Detection:
  - Recognizes vegetables in video streams.
  - Draws bounding boxes and displays the class, cluster, and confidence level.
  - Ignores low-confidence predictions.

# Folders tree
vegetable-recognition/
- |-- train/                 # Training dataset
- |-- val/                   # Validation dataset
- |-- vegetable_model.h5     # Trained model file
- |-- class_indices.json     # Mapping of class names to indices
- |-- vegetable_clusters.json # Mapping of classes to clusters
- |-- train_model.py         # Script for training the model
- |-- extract_features.py    # Script for feature extraction and clustering
- |-- recognize_video.py     # Script for real-time vegetable detection in video
- |-- README.md              # Project documentation

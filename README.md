# MobileNetV2_learning-vegetables-
MobileNetV2 learning to recognize vegetables in video flow.

This project implements a system for recognizing vegetables in video streams and assigning them to clusters based on their similarities. It includes training a neural network, clustering features, and detecting objects in video.

Features

Training a Neural Network:

Trains a MobileNetV2-based neural network to classify vegetables.
Supports data augmentation to improve robustness.

Feature Extraction and Clustering:

Extracts features from trained images.
Uses K-Means clustering to group vegetables into clusters based on feature similarity.

Real-Time Detection:

Recognizes vegetables in video streams.
Draws bounding boxes and displays the class, cluster, and confidence level.
Ignores low-confidence predictions.

vegetable-recognition/
|-- train/                 # Training dataset
|-- val/                   # Validation dataset
|-- vegetable_model.h5     # Trained model file
|-- class_indices.json     # Mapping of class names to indices
|-- vegetable_clusters.json # Mapping of classes to clusters
|-- train_model.py         # Script for training the model
|-- extract_features.py    # Script for feature extraction and clustering
|-- recognize_video.py     # Script for real-time vegetable detection in video
|-- README.md              # Project documentation

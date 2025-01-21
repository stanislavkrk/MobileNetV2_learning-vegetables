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
